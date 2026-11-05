"""Send an image over MCP to be summarized by a VLM.

Example commands:
    # OpenRouter (Claude Sonnet 5 via OpenAI API)
    python examples/mcp_image.py

    # Local Ollama (Qwen 3.8 served locally and accessed via OpenAI API)
    #   Ensure that Ollama is running and that Qwen 3.8 has been pulled:    ollama pull qwen3.8:27b
    python examples/mcp_image.py --model-id qwen3.8:27b --base-url http://localhost:11434/v1/

"""

import asyncio
import logging
from pathlib import Path
from pprint import pprint

import click
import requests
from mcp import Client
from mcp.server import MCPServer
from mcp.server.mcpserver import Image as MCPImage
from openai.types.responses import EasyInputMessageParam

from agentic_robotics.core import AgenticLoop, configure_logging
from agentic_robotics.core.filesystem import create_unique_dir
from agentic_robotics.core.foundation_models import (
    OPENAI_API_MODEL_CONFIGS,
    OpenAIModelClient,
    Transcript,
)

configure_logging()
logger = logging.getLogger(__name__)

# 1. Download an image from Wikimedia and save it to file (if not present)
data_dir = Path(__file__).parent.parent / "data"
if not data_dir.exists():
    data_dir.mkdir(parents=True)
assert data_dir.is_dir(), f"Directory {data_dir} does not exist."

image_path = data_dir / "image.jpg"

if not image_path.exists():
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Olympic_National_Park_%2822680677547%29.jpg/960px-Olympic_National_Park_%2822680677547%29.jpg"
    headers = {"User-Agent": "agentic_robotics/0.0 (https://github.com/allenai/agentic_robotics)"}
    image_response = requests.get(image_url, headers=headers)

    assert image_response.status_code == 200, (
        f"Failed to download image. Status code: {image_response.status_code}."
    )

    with open(image_path, "wb") as image_file:
        for chunk in image_response.iter_content(1024):
            image_file.write(chunk)

assert image_path.exists(), f"Image file {image_path} does not exist."

# 2. Define an MCP server to serve the image via tool call
mcp = MCPServer("GetImage")


@mcp.tool()
def get_image() -> MCPImage:
    """Retrieve an image from the MCP server."""
    return MCPImage(path=image_path)


PROMPT = (
    "You are an autonomous agent who acts solely through MCP tool calls. "
    "Identify the US national park depicted in the sole image available through your tools. "
    "In your final response, include only the name of the depicted national park."
)


@click.command()
@click.option("--model-name", default="sonnet-5", show_default=True)
def main(model_name: str) -> None:
    """Create an MCP client, call the `get_image` tool, and have a VLM describe the result."""
    asyncio.run(run(model_name))


async def run(model_name: str) -> None:
    """Run the portion of the example with an asynchronous CLI dependency."""
    assert model_name in OPENAI_API_MODEL_CONFIGS, f"Unrecognized model name: '{model_name}'."
    model_config = OPENAI_API_MODEL_CONFIGS[model_name]

    # 3. Define an OpenAI client to query a VLM
    print(f"Creating a model client for '{model_config.model_id}'...")
    model_client = OpenAIModelClient(model_config)

    out_dir = create_unique_dir(parent_dir=data_dir, dir_prefix="mcp_image-")
    transcript = Transcript(out_dir)

    # 4. Define an MCP client to call the tool
    async with Client(mcp) as mcp_client:
        capabilities = mcp_client.server_capabilities.model_dump(exclude_none=True)
        print(f"Client capabilities: {capabilities}")

        agent = AgenticLoop(mcp_client, model_client, transcript)
        initial_history = [EasyInputMessageParam(content=PROMPT, role="user")]

        transcript = await agent.enter_loop(task=PROMPT, history=initial_history, max_tool_calls=1)
        final_message = transcript.current_episode[-1]
        pprint(final_message)

        answer = final_message["content"][0]["text"]
        print(f"\n\nFinal model response: '{answer}'")


if __name__ == "__main__":
    main()
