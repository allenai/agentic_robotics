"""Put a human in the model's chair, at the `ModelClient` seam rather than at the tool boundary.

`manual_skill_driver.py` replaces the *agentic loop*: it opens its own MCP client and issues
`call_tool` directly, so nothing above the tool boundary runs. There is no transcript, no run
manifest, no budget and no preflight; the portal (`portal.py`) renders the `calls.jsonl` it writes
instead of a transcript, so a hand-driven session shows its frames and points and none of that.

This replaces the *model* instead. `ModelClient` is one async method --
`prompt(instructions, history, tools) -> History` -- so a human who returns `function_call` items
is indistinguishable to `enter_loop` from a VLM that returned the same ones. Everything above the
client therefore runs unchanged and for free: the real `Transcript`, the run manifest, preflight,
the tool-call budget, `confirm_episode_end`, and the portal.

That makes it a *control* rather than a driver, which is why the page shows the system
instructions and the "N tool calls remaining" line verbatim: the human is given exactly what the
model is given, including the same downscaled observation and the same tool schemas, so a human
session and a model session differ in the chair and nothing else.

A session needs Isaac and an MCP server behind it; see `main()` for the command. Running this
file directly needs neither Isaac nor a GPU: it checks that the page renders from a synthetic
history and that a queued batch round-trips into the `function_call` items `_filter_tool_calls`
accepts.

Three `core/` seams this file works around rather than fixes. Each is small, and none blocks a
session:

- `AgenticLoop.__init__` annotates `model_client` as `OpenAIModelClient`, the concrete class,
  while the `ModelClient` Protocol exists and is unused. Duck typing is why this file runs at all.
- `enter_loop` reads `self._model_client.config.caching` at both call sites, which is outside the
  Protocol, so implementing `ModelClient` alone raises `AttributeError` on the first turn.
  `HumanModelConfig` below exists to satisfy that read. Whether responses are cached is the
  client's business rather than the loop's, so the fix is probably to stop the loop asking.
- `mcp_conversion` puts images *into* a history and nothing takes them back out.
  `latest_image_data_uri` below is the missing inverse and belongs beside its opposite once it is
  trusted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aiohttp import web
from mcp.types import Tool as MCPTool
from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)

from agentic_robotics.core.foundation_models import create_openai_message

if TYPE_CHECKING:
    import threading

    from openai.types.responses import ResponseInputParam as History

    from agentic_robotics.adapters.toyblocks.skills.inline_oracle import InlineOracleMCPServer
    from agentic_robotics.adapters.toyblocks.skills.maniflow_skills import (
        SimThreadManiFlowServer,
    )
    from agentic_robotics.adapters.toyblocks.web_ui.interjection import InterjectionChannel
    from agentic_robotics.core.foundation_models.usage_tracking import TokenUsageSnapshot

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8082
"""Not 8000 (policy server), 8080 (viser) or 8081 (`manual_skill_driver`)."""


@dataclass(frozen=True)
class HumanModelConfig:
    """The subset of `OpenAIModelConfig` that `enter_loop` actually reads off a client.

    `ModelClient` declares only `prompt`, but `enter_loop` also reaches for
    `model_client.config.caching` at both of its call sites, so a client that implements the
    declared Protocol alone raises `AttributeError` on the first turn. Carrying a config here is
    the workaround; the fix is in `core`, and is recorded in the plan rather than taken silently.
    """

    model_id: str = "human"
    """Recorded in the run manifest, so a human session is identifiable after the fact."""

    caching: bool = False
    """Always False: there is no upstream provider to cache against."""


def latest_image_data_uri(history: History) -> str | None:
    """Find the most recent image observation in a conversation history.

    `mcp_conversion.mcp_result_to_openai_output` puts images *into* a history as
    `input_image` content blocks carrying a data URI; nothing else brings them back out, because
    nothing else reads a history rather than writing one.

    :param history: Conversation history handed to the client
    :return: The newest image's data URI, or None when the history carries no image
    """
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        # A tool result is a `function_call_output`, whose blocks hang off `output`; every other
        # item uses `content`. Reading only `content` finds the images in a hand-built history
        # and none of the ones a real episode produces, because every observation arrives as a
        # tool result.
        blocks = item.get("output") if item.get("type") == "function_call_output" else None
        if not isinstance(blocks, list):
            blocks = item.get("content")
        if not isinstance(blocks, list):
            continue
        for block in reversed(blocks):
            if isinstance(block, dict) and block.get("type") == "input_image":
                url = block.get("image_url")
                if isinstance(url, str):
                    return url
    return None


def history_text(history: History) -> list[dict[str, str]]:
    """Flatten a history into the role/text lines the page displays.

    Images are replaced by a placeholder rather than inlined a second time: the newest one is
    already shown full-size, and older ones would make the transcript pane unreadable.

    :param history: Conversation history handed to the client
    :return: One `{"role": ..., "text": ..., "image": ...}` per item, oldest first
    """
    lines: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue

        if item.get("type") == "function_call":
            lines.append(
                {"role": "tool call", "text": f"{item.get('name')}({item.get('arguments')})"}
            )
            continue
        if item.get("type") == "function_call_output":
            lines.append(
                {
                    "role": "tool result",
                    "text": _describe_output(item),
                    "image": _output_image(item),
                }
            )
            continue

        content = item.get("content")
        if isinstance(content, str):
            lines.append({"role": str(item.get("role", "?")), "text": content})
        elif isinstance(content, list):
            parts = [
                b.get("text", "") if b.get("type") in {"input_text", "output_text"} else "[image]"
                for b in content
                if isinstance(b, dict)
            ]
            lines.append({"role": str(item.get("role", "?")), "text": " ".join(parts).strip()})

    return lines


def _output_image(item: dict[str, Any]) -> str | None:
    """The image a tool result carried, if it carried one."""
    output = item.get("output")
    if isinstance(output, list):
        for block in output:
            if isinstance(block, dict) and block.get("type") == "input_image":
                url = block.get("image_url")
                if isinstance(url, str):
                    return url
    return None


def _describe_output(item: dict[str, Any]) -> str:
    """Summarize a tool result for the transcript pane."""
    output = item.get("output")
    if isinstance(output, list):
        parts = [
            b.get("text", "") if b.get("type") == "input_text" else "[image]"
            for b in output
            if isinstance(b, dict)
        ]
        return " ".join(p for p in parts if p) or "[no content]"
    return str(output)


def tool_descriptors(tools: list[MCPTool] | None) -> list[dict[str, Any]]:
    """Describe the available tools for the page, from the MCP schema the model is given.

    Built from `input_schema` rather than hardcoded, so the page cannot drift from the tool
    surface: a skill that gains an argument gains a field here with no edit.

    :param tools: The MCP tools handed to the client, as `enter_loop` lists them
    :return: One descriptor per tool, carrying its name, description and JSON schema
    """
    descriptors: list[dict[str, Any]] = []
    for tool in tools or []:
        schema = tool.input_schema or {}
        descriptors.append(
            {
                "name": tool.name,
                "title": tool.title or tool.name,
                "description": (tool.description or "").strip(),
                "schema": schema,
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
        )
    return descriptors


class HumanModelClient:
    """A `ModelClient` whose responses come from a person clicking on the observation.

    One `prompt` call is one turn: the page is handed the instructions, the history, the newest
    image and the tool schemas, and blocks until the human sends. What comes back is a batch --
    the model may return several tool calls in one turn, so the human can queue several and send
    them together, and `enter_loop` charges the budget the same way for both.
    """

    def __init__(
        self,
        config: HumanModelConfig | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        channel: InterjectionChannel | None = None,
        viser_url: str | None = None,
    ) -> None:
        """Initialize the client, without yet serving anything.

        :param config: The fields `enter_loop` reads off a client; defaults are fine
        :param host: Interface to bind, loopback by default
        :param port: Port to serve the page on
        :param channel: Optional `InterjectionChannel`, so the page's stop and reset buttons
            reach the same place the model path's do rather than growing a second mechanism
        :param viser_url: Where this run's viser server is listening, for the live inset beside
            the observation; the inset is left out when there is none
        """
        self.config = config or HumanModelConfig()
        self._host = host
        self._port = port
        self._channel = channel
        self._viser_url = viser_url or ""

        self._runner: web.AppRunner | None = None
        self._serial = 0
        self._turn: dict[str, Any] = {"state": "idle"}
        self._pending: asyncio.Future[History] | None = None
        self._usage_history: list[TokenUsageSnapshot] = []

    @property
    def usage_history(self) -> list[TokenUsageSnapshot]:
        """The token usage `AgenticLoop.take_turn` reads after each prompt; a person spends none.

        Stays empty, so the loop records no usage metadata for a human turn, the same as a model
        turn whose response carried no usage block.
        """
        return list(self._usage_history)

    async def start(self) -> None:
        """Serve the page on the loop this client is being driven from.

        Started lazily from the first `prompt` rather than in `__init__` because the aiohttp site
        must live on the agent thread's event loop, which does not exist when the client is built.
        """
        if self._runner is not None:
            return

        app = web.Application()
        app.add_routes(
            [
                web.get("/", self._index),
                web.get("/turn", self._get_turn),
                web.post("/send", self._post_send),
                web.post("/control", self._post_control),
            ]
        )
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, self._host, self._port).start()
        logger.info(f"Human model client: open http://localhost:{self._port}/ to take a turn.")

    async def close(self) -> None:
        """Stop serving the page."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def prompt(
        self, instructions: str, history: History, tools: list[MCPTool] | None = None
    ) -> History:
        """Ask the human for this turn's response, in the model's place.

        :param instructions: System-level instructions, shown verbatim
        :param history: Conversation history the response is conditioned on
        :param tools: MCP tools available this turn
        :return: The human's response as conversation items: any text they typed, then one
            `function_call` per queued tool call
        """
        await self.start()

        loop = asyncio.get_running_loop()
        self._pending = loop.create_future()
        self._serial += 1
        self._turn = {
            "state": "waiting",
            # The page redraws on a change of serial rather than by diffing the turn: two
            # consecutive turns can carry identical instructions and an identical transcript
            # tail, and diffing those silently drops the second one.
            "serial": self._serial,
            "instructions": instructions,
            "transcript": history_text(history),
            "image": latest_image_data_uri(history),
            "tools": tool_descriptors(tools),
        }

        logger.info("Waiting for a human turn; the loop is blocked until the page sends one.")
        try:
            return await self._pending
        finally:
            self._pending = None
            self._turn = {"state": "idle"}

    def build_response(self, text: str, calls: list[dict[str, Any]]) -> History:
        """Assemble one turn's conversation items from what the page sent.

        The shape is the one `_filter_tool_calls` accepts: `type="function_call"` items carrying
        `call_id`, `name` and JSON-encoded `arguments`. A turn with no calls and no text is how a
        human says "I am done", which is the same signal the model gives by calling no tool.

        :param text: Free text the human typed, if any
        :param calls: Queued calls, each `{"name": ..., "arguments": {...}}`
        :return: The conversation items to hand back to `enter_loop`
        """
        response: History = []
        if text.strip():
            response.append(create_openai_message(text.strip(), role="assistant"))

        for call in calls:
            response.append(
                ResponseFunctionToolCallParam(
                    type="function_call",
                    call_id=f"human_{uuid.uuid4().hex[:12]}",
                    name=str(call["name"]),
                    arguments=json.dumps(call.get("arguments", {})),
                )
            )
        return response

    # -------------------------------------------------------------------- HTTP
    async def _index(self, _request: web.Request) -> web.Response:
        """Serve the page, with this run's viser URL substituted in."""
        html = _PAGE.replace("{{VISER_URL}}", self._viser_url)
        return web.Response(text=html, content_type="text/html")

    async def _get_turn(self, _request: web.Request) -> web.Response:
        """Hand the page whatever the current turn is, or `idle` between turns."""
        return web.json_response(self._turn)

    async def _post_send(self, request: web.Request) -> web.Response:
        """Resolve the waiting `prompt` with the human's batch."""
        if self._pending is None or self._pending.done():
            return web.json_response({"error": "no turn is waiting"}, status=409)

        body = await request.json()
        response = self.build_response(str(body.get("text", "")), list(body.get("calls", [])))
        self._pending.set_result(response)
        return web.json_response({"ok": True, "items": len(response)})

    async def _post_control(self, request: web.Request) -> web.Response:
        """Forward stop and reset to the interjection channel the model path already uses."""
        if self._channel is None:
            return web.json_response({"error": "no interjection channel wired"}, status=409)

        action = str((await request.json()).get("action", ""))
        if action == "stop":
            self._channel.request_stop()
        elif action == "reset":
            self._channel.request_reset()
        else:
            return web.json_response({"error": f"unknown action {action!r}"}, status=400)
        return web.json_response({"ok": True})


_PAGE = """<!doctype html>
<meta charset="utf-8"><title>Human in the model's chair</title>
<style>
  :root {
    --bg: #1a1915; --panel: #211f1a; --edge: #3a372e;
    --text: #e8e6df; --dim: #a09a8c; --faint: #6b665c;
    --accent: #d97757; --user: #3d6a5b; --err: #e05d5d;
    --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--bg); color: var(--text); font: 14px/1.55 system-ui, sans-serif;
    display: grid; grid-template-columns: minmax(0,1fr) 400px;
  }
  #left { display: flex; flex-direction: column; min-width: 0; }
  #right { display: flex; flex-direction: column; border-left: 1px solid var(--edge);
           background: var(--panel); min-width: 0; }

  #status { padding: 8px 12px; border-bottom: 1px solid var(--edge); background: var(--panel);
            display: flex; align-items: center; gap: 8px; font-size: 12px; flex: none; }
  #status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--faint); flex: none; }
  #status-dot.turn { background: var(--accent); animation: pulse 1.2s infinite; }
  @keyframes pulse { 50% { opacity: 0.35; } }
  #status-text { color: var(--dim); }
  #budget { margin-left: auto; color: var(--dim); font-variant-numeric: tabular-nums; }

  #obswrap { padding: 12px; flex: none; }
  #views { display: flex; gap: 12px; align-items: flex-start; }
  #obspane, #viserpane { flex: 1 1 0; min-width: 0; }
  #obs { display: block; width: 100%; border: 1px solid var(--edge); border-radius: 6px;
         cursor: crosshair; background: #111; }

  /* The iframe matches the observation's 16:9 so the two panes end level. */
  #viserbox { position: relative; aspect-ratio: 16 / 9; border: 1px solid var(--edge);
              border-radius: 6px; overflow: hidden; background: #111; }
  #viser-frame { width: 100%; height: 100%; border: 0; display: block; }
  #viser-overlay { position: absolute; inset: 0; display: flex; align-items: center;
                   justify-content: center; text-align: center; padding: 12px;
                   background: var(--bg); color: var(--dim); font-size: 12px;
                   white-space: pre-wrap; }
  #viser-overlay.hidden { display: none; }
  #viser-cap { margin-top: 6px; font-family: var(--mono); font-size: 12px; color: var(--dim);
               display: flex; gap: 8px; }
  #viser-cap a { color: var(--accent); text-decoration: none; margin-left: auto; }
  #where { margin-top: 6px; font-family: var(--mono); font-size: 12px; color: var(--dim); }
  #where b { color: var(--accent); font-weight: normal; }

  h4 { margin: 0 0 6px; font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
       color: var(--faint); font-weight: 600; }
  #scroll { flex: 1; overflow-y: auto; padding: 0 12px 12px; min-height: 0; }
  pre#instr { background: var(--panel); border: 1px solid var(--edge); border-radius: 6px;
              padding: 8px 10px; margin: 0 0 14px; font-size: 12px; color: var(--dim);
              white-space: pre-wrap; overflow-wrap: break-word; font-family: var(--mono); }

  .msg { margin: 0 0 12px; white-space: pre-wrap; overflow-wrap: break-word; }
  .msg .role { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
               color: var(--faint); margin-bottom: 2px; }
  .msg.user { border-left: 3px solid var(--user); padding-left: 10px; }
  .msg.assistant { border-left: 3px solid var(--accent); padding-left: 10px; }
  .msg.system { color: var(--faint); font-size: 12px; }
  .msg.toolcall { border-left: 3px solid var(--edge); padding-left: 10px;
                  font-family: var(--mono); font-size: 12px; color: var(--dim); }
  .msg.toolresult { margin-left: 14px; font-size: 13px; color: var(--dim); }
  .msg.toolresult img { display: block; max-width: 240px; margin-top: 6px; cursor: zoom-in;
                        border: 1px solid var(--edge); border-radius: 4px; }

  #panel { padding: 12px; overflow-y: auto; flex: 1; min-height: 0; }
  .tool { display: block; width: 100%; text-align: left; margin: 0 0 6px;
          background: var(--bg); color: var(--text); border: 1px solid var(--edge);
          border-radius: 6px; padding: 7px 10px; cursor: pointer; font: inherit; }
  .tool:hover { border-color: var(--accent); }
  .tool .name { font-family: var(--mono); font-size: 13px; }
  .tool .desc { display: block; color: var(--faint); font-size: 11px; margin-top: 2px;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  .q { display: flex; align-items: center; gap: 8px; background: var(--bg);
       border: 1px solid var(--edge); border-left: 3px solid var(--accent); border-radius: 6px;
       padding: 6px 8px; margin: 0 0 6px; font-family: var(--mono); font-size: 12px; }
  .q span { flex: 1; overflow: hidden; text-overflow: ellipsis; }
  .q button { background: transparent; border: 0; color: var(--faint); cursor: pointer;
              font-size: 14px; padding: 0 4px; }
  .q button:hover { color: var(--err); }
  #empty { color: var(--faint); font-size: 12px; font-style: italic; margin-bottom: 6px; }

  #composer { border-top: 1px solid var(--edge); padding: 10px 12px; flex: none; }
  #text { width: 100%; resize: none; background: var(--bg); color: var(--text);
          border: 1px solid var(--edge); border-radius: 6px; padding: 8px; font: inherit; }
  #text:focus, .tool:focus { outline: 1px solid var(--accent); }
  #buttons { display: flex; gap: 6px; margin-top: 8px; }
  button.act { background: var(--bg); color: var(--text); border: 1px solid var(--edge);
               border-radius: 6px; padding: 8px 14px; cursor: pointer; font: inherit; }
  button.act:hover { border-color: var(--accent); }
  #send { border-color: var(--accent); color: var(--accent); flex: 1; }
  #send:disabled { opacity: .4; cursor: default; border-color: var(--edge); color: var(--faint); }
  #stop:hover, #reset:hover { background: #3d2320; border-color: var(--err); color: var(--err); }

  #lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.85); display: none;
              align-items: center; justify-content: center; cursor: zoom-out; z-index: 10; }
  #lightbox img { max-width: 94%; max-height: 94%; }
</style>
<div id="left">
  <div id="status">
    <span id="status-dot"></span><span id="status-text">connecting...</span>
    <span id="budget"></span>
  </div>
  <div id="obswrap">
    <div id="views">
      <div id="obspane">
        <img id="obs" alt="the observation the model would be shown">
        <div id="where">click the image to set a pixel</div>
      </div>
      <div id="viserpane">
        <div id="viserbox">
          <iframe id="viser-frame" title="live scene"></iframe>
          <div id="viser-overlay">connecting to viser…</div>
        </div>
        <div id="viser-cap"><span>live scene</span><a id="viser-link" target="_blank"
           rel="noopener">open ↗</a></div>
      </div>
    </div>
  </div>
  <div id="scroll">
    <h4>instructions</h4><pre id="instr"></pre>
    <h4>conversation</h4><div id="hist"></div>
  </div>
</div>
<div id="right">
  <div id="panel">
    <h4>tools</h4><div id="tools"></div>
    <h4 style="margin-top:16px">queued batch</h4>
    <div id="empty">nothing queued &mdash; sending an empty batch ends the episode</div>
    <div id="queue"></div>
  </div>
  <div id="composer">
    <textarea id="text" rows="3" placeholder="message (optional)"></textarea>
    <div id="buttons">
      <button class="act" id="send">send turn</button>
      <button class="act" id="stop">stop</button>
      <button class="act" id="reset">reset</button>
    </div>
  </div>
</div>
<div id="lightbox"><img></div>
<script>
let turn=null, pixel=null, queue=[];
const $=(i)=>document.getElementById(i);

// The live scene, beside the observation. Simpler than web_ui/index.html's version: this page
// never builds a scene, so there is no loading overlay and no re-pointing -- one URL, fixed for
// the run. Probed with a bare websocket because viser's HTTP server is a websocket server and
// websockets are not subject to CORS, so the iframe only loads once the port actually answers
// and an unreachable viser says which URL is unreachable instead of showing a blank pane.
const CONFIGURED_VISER_URL = "{{VISER_URL}}";
if (!CONFIGURED_VISER_URL) {
  $("viserpane").remove();
} else {
  const VISER_URL = ["localhost","127.0.0.1"].includes(location.hostname)
    ? CONFIGURED_VISER_URL
    : CONFIGURED_VISER_URL.replace("//localhost","//"+location.hostname)
                          .replace("//127.0.0.1","//"+location.hostname);
  $("viser-link").href = VISER_URL;
  const probeViser = () => {
    let settled=false, probe=null;
    const fail=(why)=>{
      if(settled) return;
      settled=true;
      if(probe) try { probe.close(); } catch(e) {}
      $("viser-overlay").classList.remove("hidden");
      $("viser-overlay").textContent =
        "viser is not reachable at "+VISER_URL+" from this browser ("+why+") — retrying…\\n\\n"+
        "This page came from "+location.host+". If you port-forwarded "+(location.port||"80")+
        " to get here, forward 8080 the same way.";
      setTimeout(probeViser,3000);
    };
    try { probe=new WebSocket(VISER_URL.replace(/^http/,"ws")); }
    catch(e) { fail(e.name); return; }
    const timer=setTimeout(()=>fail("connection is hanging"),4000);
    probe.onopen=()=>{
      settled=true; clearTimeout(timer); probe.close();
      $("viser-overlay").classList.add("hidden");
      // Always (re)load: a viser server restarted under an open iframe leaves the old client
      // showing its last frame forever, so a fresh probe means a fresh iframe.
      $("viser-frame").src = VISER_URL;
    };
    probe.onerror=()=>{ clearTimeout(timer); fail("connection failed"); };
  };
  probeViser();
}
const CLS={"user":"user","assistant":"assistant","system":"system",
           "tool call":"toolcall","tool result":"toolresult"};

$("obs").addEventListener("click",(e)=>{
  const r=e.target.getBoundingClientRect();
  pixel={u:Math.round((e.clientX-r.left)/r.width*e.target.naturalWidth),
         v:Math.round((e.clientY-r.top)/r.height*e.target.naturalHeight)};
  $("where").innerHTML=`pixel <b>(${pixel.u}, ${pixel.v})</b> of `
    +`${e.target.naturalWidth}x${e.target.naturalHeight}`;
});
$("lightbox").onclick=()=>$("lightbox").style.display="none";
function zoom(src){ $("lightbox").querySelector("img").src=src;
                    $("lightbox").style.display="flex"; }

function renderTools(){
  $("tools").innerHTML="";
  (turn.tools||[]).forEach(t=>{
    const b=document.createElement("button"); b.className="tool"; b.title=t.description;
    b.innerHTML=`<span class="name">${t.name}</span>`;
    if(t.description){ const d=document.createElement("span"); d.className="desc";
                       d.textContent=t.description; b.appendChild(d); }
    b.onclick=()=>{
      const args={};
      for(const [k,spec] of Object.entries(t.properties||{})){
        if(spec.$ref||spec.type==="object"){
          if(!pixel){ $("where").innerHTML="<b>click the image first</b>"; return; }
          args[k]=pixel;
        } else {
          const v=window.prompt(`${t.name}.${k}`
            +(spec.description?` \u2014 ${spec.description}`:""));
          if(v===null) return;
          args[k]=(spec.type==="number"||spec.type==="integer")?Number(v):v;
        }
      }
      queue.push({name:t.name,arguments:args}); renderQueue();
    };
    $("tools").appendChild(b);
  });
}
function renderQueue(){
  $("queue").innerHTML=""; $("empty").style.display=queue.length?"none":"block";
  queue.forEach((c,i)=>{
    const d=document.createElement("div"); d.className="q";
    const s=document.createElement("span");
    s.textContent=`${i+1}. ${c.name}(${JSON.stringify(c.arguments)})`;
    const x=document.createElement("button"); x.textContent="\u00d7"; x.title="remove";
    x.onclick=()=>{queue.splice(i,1);renderQueue();};
    d.append(s,x); $("queue").appendChild(d);
  });
}
function renderTurn(t){
  turn=t; queue=[]; pixel=null;
  $("status-dot").className="turn"; $("status-text").textContent="your turn";
  $("instr").textContent=t.instructions||"";
  $("where").textContent="click the image to set a pixel";
  if(t.image){ $("obs").src=t.image; $("obs").style.display="block"; }
  else { $("obs").removeAttribute("src"); $("obs").style.display="none";
         $("where").textContent="no image in this turn's history yet"; }
  $("hist").innerHTML="";
  (t.transcript||[]).forEach(l=>{
    const d=document.createElement("div"); d.className="msg "+(CLS[l.role]||"system");
    const r=document.createElement("div"); r.className="role"; r.textContent=l.role;
    d.appendChild(r); d.append(l.text||"");
    if(l.image){ const im=document.createElement("img"); im.src=l.image;
                 im.onclick=()=>zoom(l.image); d.appendChild(im); }
    $("hist").appendChild(d);
  });
  const m=/(\\d+) tool calls remaining/.exec(JSON.stringify(t.transcript||[]));
  $("budget").textContent=m?`${m[1]} calls left`:"";
  renderTools(); renderQueue(); $("send").disabled=false;
  $("scroll").scrollTop=$("scroll").scrollHeight;
}
async function poll(){
  let t; try { t=await (await fetch("/turn")).json(); }
  catch { $("status-dot").className=""; $("status-text").textContent="server unreachable"; return; }
  if(t.state==="idle"){
    $("status-dot").className=""; $("status-text").textContent="the loop is busy";
    $("send").disabled=true; turn=null;
  } else if(!turn || t.serial!==turn.serial) renderTurn(t);
}
$("send").onclick=async()=>{
  $("send").disabled=true;
  const r=await fetch("/send",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({text:$("text").value,calls:queue})});
  if(r.ok){ $("text").value=""; queue=[]; renderQueue(); turn=null;
            $("status-dot").className=""; $("status-text").textContent="sent"; }
  else { $("send").disabled=false; $("status-text").textContent=(await r.json()).error; }
};
for(const a of ["stop","reset"]) $(a).onclick=()=>fetch("/control",{method:"POST",
  headers:{"Content-Type":"application/json"},body:JSON.stringify({action:a})});
setInterval(poll,700); poll();
</script>
"""


def human_agentic_thread(
    server: InlineOracleMCPServer | SimThreadManiFlowServer,
    finished: threading.Event,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
) -> None:
    """Drop-in for `toyblocks_runner._agentic_thread`, with a human in the model's chair.

    Same signature and the same manual event loop as `web_ui`'s `agentic_web_thread`, for the
    same reason: Kit replaces `asyncio.run` on the main thread, so the loop is driven by hand on
    this one. Pass it to `runner_bootstrap.run_episode(agent_thread=...)` and every mode that
    already exists keeps working around it.

    :param server: MCP server providing the tools
    :param finished: `threading.Event` set on the way out, releasing the simulator's thread
    :param host: Interface to bind
    :param port: Port to serve the page on
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_human_episode(server, host=host, port=port))
    except Exception:
        logger.exception("Human-driven episode failed.")
    finally:
        loop.close()
        finished.set()


async def _run_human_episode(
    server: InlineOracleMCPServer | SimThreadManiFlowServer, *, host: str, port: int
) -> None:
    """Run one episode with a `HumanModelClient` where the model client normally goes.

    Deliberately a copy of `toyblocks_runner._run_agentic_loop` with one line changed, which is
    the whole claim this file makes: the loop, the transcript and the budget do not know or care
    that the responses came from a person.

    :param server: MCP server providing the tools
    :param host: Interface to bind
    :param port: Port to serve the page on
    """
    from mcp import Client as MCPClient
    from openai.types.responses import EasyInputMessageParam

    from agentic_robotics.core.agentic_loop import AgenticLoop
    from agentic_robotics.core.foundation_models import Transcript

    # `viewer_url` is asked of the server rather than assumed, because viser silently takes the
    # next free port when its default is busy.
    model_client = HumanModelClient(  # <- the only change
        host=host, port=port, viser_url=server.viewer_url()
    )
    transcript = Transcript(out_dir=server.config.output_dir)

    async with MCPClient(server.mcp) as mcp_client:
        agent = AgenticLoop(
            mcp_client,
            model_client,
            transcript,
            metadata_callback=lambda: {"frame_count": server.frame_count},
        )
        task_prompt = server.config.task_config.prompt

        try:
            await agent.enter_loop(
                task=task_prompt,
                history=[EasyInputMessageParam(content=task_prompt, role="user")],
                max_tool_calls=server.config.task_config.max_tool_calls,
            )
        finally:
            await model_client.close()


def _selftest(port: int) -> None:
    """Serve one synthetic turn, so the page can be exercised without Isaac or a GPU.

    Checks the two things that do not need a simulator: that a history renders (instructions,
    transcript, image, tool schemas), and that a queued batch round-trips into the
    `function_call` items `_filter_tool_calls` accepts.

    :param port: Port to serve the page on
    """
    from agentic_robotics.core.agentic_loop import is_tool_call

    # A 1x1 red PNG, so the image path is exercised without a camera.
    red_dot = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
        "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    history: History = [
        create_openai_message("Put the R on the red block.", role="user"),
        {"role": "user", "content": [{"type": "input_image", "image_url": red_dot}]},
    ]
    tools = [
        MCPTool(
            name="pick",
            description="Pick up the block at the given pixel.",
            inputSchema={
                "type": "object",
                "properties": {"pixel": {"type": "object", "description": "raw (u,v)"}},
                "required": ["pixel"],
            },
        )
    ]

    client = HumanModelClient(port=port)
    assert latest_image_data_uri(history) == red_dot, "image extraction failed"
    assert tool_descriptors(tools)[0]["name"] == "pick", "tool descriptor failed"

    built = client.build_response(
        "looks fine", [{"name": "pick", "arguments": {"pixel": {"u": 3, "v": 4}}}]
    )
    calls = [item for item in built if is_tool_call(item)]
    assert len(calls) == 1 and calls[0]["name"] == "pick", "batch did not round-trip"
    assert json.loads(calls[0]["arguments"]) == {"pixel": {"u": 3, "v": 4}}, "arguments mangled"
    print("selftest: image extraction, tool descriptors and batch round-trip all OK")

    async def serve_one_turn() -> None:
        response = await client.prompt(
            "(selftest) same instructions the model would see", history, tools
        )
        print(f"page returned {len(response)} item(s): {json.dumps(response, indent=2)[:400]}")
        await client.close()

    print(f"serving a synthetic turn at http://localhost:{port}/ -- Ctrl-C to stop")
    asyncio.run(serve_one_turn())


def main() -> None:
    """Serve one synthetic turn, so the page can be exercised without Isaac.

    The real path is a mode of `agentic-robotics`, not a second entry point here:

        agentic-robotics human --task r_on_red

    `agentic-robotics human` boots the scene exactly as `episode` does and passes
    :func:`human_agentic_thread` in place of the default agent thread, so the human and the model
    share one boot path rather than two that can drift.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    _selftest(args.port)


if __name__ == "__main__":
    main()
