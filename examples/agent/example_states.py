"""Probe: write the two example initial states for the experimentation phases.

Drives a served `agentic-robotics pool` on ai2_stacked_09 (booted with its visibility preflight on, so its
reset state is one every block can be pointed at) over MCP, no agent. Saves that reset state as
the cluttered example, then writes the ai2_easy base scene's three blocks (A, I, 2 at their
ai2_easy poses) into the same package, distractors stowed, reads it back and saves it as the
simple example. Images beside each.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, "examples/agent")
from flip_trials import call_tool, read_state

from agentic_robotics.core.states import SimState

URL, EASY, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)


async def main():
    stacked = await read_state(URL)
    print("ai2_stacked reset state:", len(stacked.objects), "blocks", flush=True)
    (OUT / "ai2_stacked.json").write_text(stacked.model_dump_json(indent=1) + "\n")
    _, image = await call_tool(URL, "get_image_observation", {})
    if image:
        (OUT / "ai2_stacked.png").write_bytes(image)
    easy = SimState.model_validate_json(EASY.read_text())
    simple = stacked.model_copy(deep=True)
    simple.robot_base_pose = easy.robot_base_pose
    simple.robot_configuration = easy.robot_configuration
    simple.objects = [o for o in easy.objects]
    text, image = await call_tool(URL, "set_sim_state", {"state": simple.model_dump()})
    back = await read_state(URL)
    print(
        "three_blocks written and read back:",
        [(o.name, round(o.pose.position.x, 2), round(o.pose.position.y, 2)) for o in back.objects],
        "|",
        (text or "")[:80],
        flush=True,
    )
    (OUT / "three_blocks.json").write_text(back.model_dump_json(indent=1) + "\n")
    if image:
        (OUT / "three_blocks.png").write_bytes(image)
    await call_tool(URL, "set_sim_state", {"state": stacked.model_dump()})
    print("done", flush=True)


asyncio.run(main())
