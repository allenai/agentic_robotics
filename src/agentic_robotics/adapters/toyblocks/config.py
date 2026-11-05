"""The configuration schema for the ToyBlocks adapter: the task table, the prompts, and the
dataclasses a run's configuration is made of. Values come from an experiment file; see
`experiment.py`."""

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from agentic_robotics.adapters.toyblocks.maniflow_config import ManiFlowConfig
from agentic_robotics.adapters.toyblocks.release.search_spell_families import (
    COLLECTIONS as SEARCH_SPELL_COLLECTIONS,
)
from agentic_robotics.adapters.toyblocks.release.search_spell_families import (
    EXAMPLE_WORDS as SEARCH_SPELL_EXAMPLE_WORDS,
)
from agentic_robotics.adapters.toyblocks.release.search_spell_families import (
    LETTER_BLOCKS_RANGE as SEARCH_SPELL_LETTER_BLOCKS_RANGE,
)
from agentic_robotics.adapters.toyblocks.release.search_spell_families import (
    WORDS as SEARCH_SPELL_WORDS,
)
from agentic_robotics.adapters.toyblocks.release.search_spell_families import (
    task_key as search_spell_task_key,
)
from agentic_robotics.core.foundation_models import OpenAIModelConfig
from agentic_robotics.core.images import Resolution


class SkillBackend(StrEnum):
    """Enumeration over backends implementing skills for the ToyBlocks domain."""

    MANIFLOW = "maniflow"
    """Learned ManiFlow policies, reached over a websocket."""

    ORACLE = "oracle"
    """Scripted ToyBlocks vecgen skills (cuRobo v2 planning) on privileged simulator state."""


PROBLEMS_DIR = Path(__file__).resolve().parents[4] / "data/problems"


@dataclass(frozen=True)
class TaskConfig:
    """Configuration specifying the prompt and PDDL file for a task."""

    name: str
    """Human-readable name of the task."""

    task_slug: str
    """Slug used by autogen for the task's environment."""

    prompt: str
    """Text used to prompt the agent at the beginning of the task."""

    max_tool_calls: int
    """Task-specific budget of MCP calls the agent is permitted to make."""

    glyph_library: str | None = None
    """Glyph asset library the task's packages are built against (a name from ToyBlocks's
    `glyph_blocks.GLYPH_LIBRARIES`); None leaves the choice to the runner's default."""

    @property
    def env_name(self) -> str:
        """Gymnasium environment ID registered by autogen."""
        return f"{self.task_slug}-v0"

    @property
    def pddl_file(self) -> Path:
        """Filepath to the PDDL file for the task."""
        return PROBLEMS_DIR / f"{self.task_slug}.pddl"


IMAGE_OBS_WIDTH = 640
IMAGE_OBS_HEIGHT = 360

PROMPT_PREFIX = (
    "You are the brain of an autonomous robot. You act solely through MCP tool calls, "
    "each corresponding to a robot motor skill or perception query. Your goal is to "
    "complete each task while minimizing the number of robot skills executed. Some "
    "tasks may require long-horizon reasoning. Therefore, given a task, be thorough "
    "and form a coherent strategy before acting.\n\n"
    "Complete the following task using the available MCP tools:\n\n"
)


AI2_STACK_PROMPT = (
    "There are various letters and symbols on the faces of the blocks on the table. "
    "Your task is to use the blocks to spell the word 'AI2' in a tower, with 'A' "
    "on top and '2' on bottom, with all three letters upright and facing the camera. "
)


LED_STACK_PROMPT = (
    "Stack the three blocks to spell 'LED' from top to bottom, "
    "with each letter upright and facing the camera. "
)


PIXEL_COORDINATE_PROMPT = (
    "Every pixel you give to a tool is a raw (u,v) coordinate in the standard image "
    f"observation, which is {IMAGE_OBS_WIDTH} pixels wide and {IMAGE_OBS_HEIGHT} pixels tall. "
    f"u runs from 0 to {IMAGE_OBS_WIDTH - 1} left to right, and v runs from 0 to "
    f"{IMAGE_OBS_HEIGHT - 1} top to bottom. Do not use pixel coordinates from zoomed images.\n\n"
)


PROMPT_SUFFIX = PIXEL_COORDINATE_PROMPT + (
    "When you are done, you may terminate the episode by returning zero tool calls. "
    "Double-check before exiting. If you cannot complete the task, it is better to exit "
    "early than waste tool calls and other resources achieving an incorrect final state."
)

TOYBLOCKS_TASKS: dict[str, TaskConfig] = {
    "wasp_tower": TaskConfig(
        name="Wasp Tower",
        task_slug="wasp_tower",
        prompt=PROMPT_PREFIX + "Build a wasp tower using the blocks on the table. " + PROMPT_SUFFIX,
        max_tool_calls=20,
    ),
    "r_on_red": TaskConfig(
        name="R on Red",
        task_slug="r_on_red",
        prompt=PROMPT_PREFIX
        + (
            "The letter block has the letter 'R' on one of its faces. Your task is to place "
            "that block on the red block such that the 'R' is upright and facing the camera. "
        )
        + PROMPT_SUFFIX,
        max_tool_calls=25,
    ),
    "find_r": TaskConfig(
        name="Find R",
        task_slug="find_r",
        prompt=PROMPT_PREFIX
        + (
            "There are various letters and symbols on the faces of the blocks on the table. "
            "Find the block with an 'R' and place it with the 'R' upright and facing the camera. "
        )
        + PROMPT_SUFFIX,
        max_tool_calls=10,
    ),
    "ai2_stacked": TaskConfig(
        name="Ai2 Stack w/ Pre-Stacked Distractors",
        task_slug="ai2_stacked",
        prompt=PROMPT_PREFIX + AI2_STACK_PROMPT + PROMPT_SUFFIX,
        max_tool_calls=80,
    ),
    "ai2_letters": TaskConfig(
        name="Ai2 Stack w/ Letter Distractor Blocks",
        task_slug="ai2_letters",
        prompt=PROMPT_PREFIX + AI2_STACK_PROMPT + PROMPT_SUFFIX,
        max_tool_calls=60,
    ),
    "ai2_easy": TaskConfig(
        name="Ai2 Stack, Easy",
        task_slug="ai2_easy",
        prompt=PROMPT_PREFIX + AI2_STACK_PROMPT + PROMPT_SUFFIX,
        max_tool_calls=30,
    ),
    "ai2_medium": TaskConfig(
        name="Ai2 Stack, Medium",
        task_slug="ai2_medium",
        prompt=PROMPT_PREFIX + AI2_STACK_PROMPT + PROMPT_SUFFIX,
        max_tool_calls=40,
    ),
}


def construct_search_and_spell_prompt(word: str) -> str:
    """Construct a prompt for the 'Search & Spell' task using the given word."""
    return (
        PROMPT_PREFIX
        + (
            "There are various letters and symbols on the faces of the blocks on the table. "
            f"Your task is to spell the word '{word.upper()}' in a tower, with its first "
            "letter on the top block and its last letter on the bottom block, with all "
            "letters in the word upright and facing the camera. You may use any mix of uppercase "
            "and lowercase letters, as long as the word is spelled correctly.\n\n"
            "You will be given a budget of tool calls for your motor skills. Once this "
            "budget runs out, the episode will be terminated.\n\n"
        )
        + PIXEL_COORDINATE_PROMPT
        + (
            "When you are done, you may terminate the episode by returning zero tool calls, "
            "and include the substring `[[SUCCESS]]` in your final conversation message. "
            "If you have exhaustively searched all blocks and are certain that your task is "
            "impossible, terminate the episode and instead include the substring `[[IMPOSSIBLE]]`. "
            "Double-check before exiting."
        )
    )


SEARCH_SPELL_MOTOR_CALLS_PER_LETTER_BLOCK = 36
"""Motor skill cutoff multiplier used for Search & Spell episodes: motor calls per letter block
(216 for six, 288 for eight), every motor skill counted. A scene needs about 32 effective
manipulation calls; at the agent's measured failure rate of about 37% that is about 51 expected
calls; the agent returns the arm home after most skills to clear the camera, about 0.65 to 0.76
`return_home` calls per manipulation call, which brings the expectation to about 90. The rate is
set at about 2.4x that expectation rather than at a margin over it because the agent sees its
remaining budget and stops short of it: a cap near the expectation decides when the agent gives
up, and the number of calls an episode takes is itself a measurement. The cap is a guard against
runaway episodes, not a condition of the experiment."""

SEARCH_SPELL_TOTAL_CALLS_FACTOR = 2.5
"""The ceiling on calls of any kind, perception included, as a multiple of the motor cutoff: a
guard against perception loops, not the cutoff (540 and 720)."""

SEARCH_SPELL_DECLARATIONS: dict[str, str] = {
    "[[SUCCESS]]": "success",
    "[[IMPOSSIBLE]]": "impossible",
}
"""The substrings the prompt asks the agent to put in its final message, and what each declares."""

# The Search & Spell evaluation tasks: one key per target word and block collection, each owning
# twelve instances under `data/problems/<key>/` (see `release.search_spell_families`); and the
# example tasks an agent experiments on, one key per example word and collection. The motor
# cutoff here is the fallback; a Search & Spell episode takes its cutoff from the instance's
# family (`leap.evaluate.search_spell_episode_caps`).
TOYBLOCKS_TASKS.update(
    {
        search_spell_task_key(word, collection): TaskConfig(
            name=f"Search & Spell: {word} ({collection} blocks)",
            task_slug=search_spell_task_key(word, collection),
            prompt=construct_search_and_spell_prompt(word),
            max_tool_calls=SEARCH_SPELL_MOTOR_CALLS_PER_LETTER_BLOCK
            * SEARCH_SPELL_LETTER_BLOCKS_RANGE[1],
            glyph_library=library,
        )
        for word in (*SEARCH_SPELL_WORDS, *SEARCH_SPELL_EXAMPLE_WORDS)
        for collection, library in SEARCH_SPELL_COLLECTIONS.items()
    }
)


@dataclass(frozen=True)
class OracleConfig:
    """Configuration for the ground-truth (oracle) skills."""

    settle_steps: int = 10
    """Ticks simulated to settle the arm after it returns home."""

    pointing_error_ratio: float = 1.2
    """A pointing pixel is rejected if it's off by more than this ratio of the block's projected
    radius."""

    skill_timeout_steps: int = 1500
    """Ticks (25 Hz) one skill may take before it is abandoned as failed."""

    release_settle_steps: int = 25
    """Ticks simulated after a place, flip, rotate or gripper command, so the block comes to rest
    before the scene is observed (about 1 s)."""

    return_home_steps: int = 75
    """Ticks over which `return_home` ramps the arm joints to their home configuration."""


DEFAULT_RUNS_ROOT = Path(
    os.environ.get("AGENTIC_ROBOTICS_RUNS_DIR", Path(__file__).resolve().parents[4] / "runs")
)
"""Parent directory under which every run's output is saved: `$AGENTIC_ROBOTICS_RUNS_DIR`, or
`runs/` in this repo."""


@dataclass(frozen=True)
class ToyBlocksConfig:
    """One run's configuration for the ToyBlocks MCP adapter.

    Built by `experiment.build_config` from an experiment's settings, the task and the run
    directory; immutable, passed to the components that read it, and written to the run directory
    as ``config.json``.
    """

    skill_backend: SkillBackend
    """Which implementation of the skills the runner boots a server for."""

    model_config: OpenAIModelConfig
    task_config: TaskConfig

    maniflow: ManiFlowConfig
    """Configuration for the ManiFlow skills."""

    output_dir: Path
    """The run's directory."""

    image_obs_width: int = IMAGE_OBS_WIDTH
    image_obs_height: int = IMAGE_OBS_HEIGHT

    viewer: bool = True

    action_horizon: int = 25

    oracle: OracleConfig = field(default_factory=OracleConfig)
    """Configuration for the ground-truth (oracle) skills."""

    camera_name: str = "front_camera"
    """Name of the camera used to capture images of the scene for the exported video."""

    rng_seed: int = 1612

    output_root_dir: Path = DEFAULT_RUNS_ROOT
    """Parent directory of the run directory."""

    experiment: str = "default"
    """Name of the experiment the settings came from."""

    @property
    def observation_resolution(self) -> Resolution:
        """Retrieve the configured resolution used for image observations passed to the agent."""
        return Resolution(height=self.image_obs_height, width=self.image_obs_width)

    def validate(self) -> None:
        """Validate that the configured values are valid (raises if not)."""
        if not self.output_root_dir.is_dir():
            raise RuntimeError(f"Output root directory not found: {self.output_root_dir.resolve()}")
        if not self.output_dir.is_dir():
            raise RuntimeError(f"Output directory not found: {self.output_dir.resolve()}")
