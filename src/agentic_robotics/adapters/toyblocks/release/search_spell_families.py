"""The Search & Spell task families as data: the specification's constants, the block face maps,
and the per-task `families.json` that `drivers/search_spell_env.py` writes and that
`search_spell_scoring` and ``agentic-robotics-spell-build` (drivers/search_spell_build.py)` read.

The developer's task specification is the source of every number here. Units, in its words: a
**task family** fixes the target word, the
block inventory (which concrete blocks, how many of them letter blocks), which blocks begin
stacked, and every block's abstract orientation; a **task instance** is one continuous sampling
of a family (poses, block sizes, table height, friction) by Stage-1 codegen under the training
data's settings. Each instance is its own PDDL file and so its own autogen package
(`task_variants` maps one PDDL to one package), which is what lets every instance carry its own
scene profile in env slot 0, the only slot an episode runs in.

Layout, under `data/problems/<task key>/`:

    spell_<word>_<collection>/                     one task key per word and block collection
        spell_<word>_<collection>_<k>_<feasibility>_<j>.pddl
                                                   inventory k (6+k letter blocks), `feasible` or
                                                   `infeasible`, instance j
        families.json                              every family of the task, keyed by family slug

Every family hides its target letters at the start (the specification's visibility rule: a face
is visible on the front of a block, or on top of a block with nothing on it). An infeasible
family is its feasible counterpart with one target letter absent: the block or blocks carrying
that letter are replaced, in the same slots of the same stack layout, by blocks carrying no
letter of the word, so the word cannot be spelled and everything else matches. Within one word
and inventory, the patterned and unpatterned families share their letter-block count, their
color cubes and their stack layout over numbered block slots (targets fill the first slots).

Block names are PDDL object names: `block_<letter>` for the patterned collection
(`train_blocks_large`), `block_NNNNN` for the unpatterned one (`eval_blocks_36_0`), and the
training data's six color cubes. The face maps under `data/block_faces/` come from
`scripts/decode_block_faces.py`.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENTIC_ROBOTICS_ROOT = Path(__file__).resolve().parents[5]
"""The checkout: this file is `src/agentic_robotics/adapters/toyblocks/release/` below it."""
PROBLEMS_DIR = AGENTIC_ROBOTICS_ROOT / "data" / "problems"
"""The same directory `config.PROBLEMS_DIR` names; not imported from there because `config`
imports this module for the task keys."""
BLOCK_FACES_DIR = AGENTIC_ROBOTICS_ROOT / "data" / "block_faces"

WORDS: tuple[str, ...] = ("AUK", "EMU", "HEN", "JAY", "OWL")
"""The specification's five target words; every one is spellable from three distinct blocks of
either collection."""

EXAMPLE_WORDS: tuple[str, ...] = ("MOA",)
"""The words of the example families an agent experiments on before the suite: outside `WORDS`,
so no suite instance shares a word with them, and outside `task_keys()`, so the suite never
runs them. Their families are feasible only and run from six to ten letter blocks."""

COLLECTIONS: dict[str, str] = {
    "patterned": "train_blocks_large",
    "unpatterned": "eval_blocks_36_0",
}
"""Collection name, as it appears in task keys, to the glyph library its packages are built
against."""

FEASIBILITIES: tuple[str, ...] = ("feasible", "infeasible")
"""`feasible`: the scene holds every letter of the word (in the unpatterned collection, exactly one
block carrying its uppercase and one its lowercase). `infeasible`: one target letter is absent in
both cases, and the agent's correct outcome is to report the task impossible."""

SCENE_BLOCKS = 10
LETTER_BLOCKS_RANGE = (6, 8)
"""Letter/digit blocks per scene; inventory k of a word has `LETTER_BLOCKS_RANGE[0] + k`. The rest
are color cubes."""
STACKS_RANGE = (1, 5)
"""Two-block stacks per scene, uniform; every other block is alone on the table."""
MIN_SEPARATION_M = 0.07
"""Center-to-center, in the table plane, between blocks not in the same stack."""

BLOCK_SIZE_RANGE = (0.035, 0.05)
"""Block edge, uniform per package, written into every PDDL for codegen. The training data's
range is 3-5 cm; the evaluation tasks keep its upper 1.5 cm because the ManiFlow flip and rotate
skills fail about half the time on blocks under 3.5 cm and about one time in ten above it, and
the tasks are meant to measure the agent, not the policy's lower size limit."""
TABLE_HEIGHT_RANGE = (-0.05, 0.05)
FRICTION_RANGE = (0.5, 0.5)
"""The training data's settings (`oct06_stack3_1000x32`,
`run_datagen_pipeline_config_20261009_112203.json`), written into every PDDL for codegen."""

COLOR_CUBES: dict[str, tuple[str, str]] = {
    "blue": ("block_blue", "block_blue"),
    "yellow": ("block_yellow", "block_yellow"),
    "orange": ("block_orange", "block_orange"),
    "green": ("block_green", "block_green"),
    "red": ("block_red1", "block_red"),
    "purple": ("block_purple", "block_purple"),
}
"""Color to (PDDL object name, PDDL type): the six cubes the training data mixes in
(`scripts/task_gen/stack_three.py` at the training commit)."""

LETTER_TYPE = "block_letter"

SLUG_PATTERN = re.compile(
    r"^(?P<task>spell_(?P<word>[a-z]{3})_(?P<collection>patterned|unpatterned))"
    r"_(?P<inventory>\d+)_(?P<feasibility>feasible|infeasible)(?:_(?P<instance>\d+))?$"
)


@dataclass(frozen=True)
class LetterFace:
    """A letter on one body face of a block, and which body axis its top points along."""

    face: str
    glyph: str
    up: str


@dataclass(frozen=True)
class BlockFaces:
    """Every glyph on a block, from `data/block_faces/<library>.json`."""

    name: str
    usd: str
    faces: dict[str, dict[str, str]]

    def letters(self) -> list[LetterFace]:
        """The faces carrying a letter A-Z or a-z, in face order."""
        return [
            LetterFace(face, info["glyph"], info["up"])
            for face, info in self.faces.items()
            if info["type"] == "character" and info["glyph"] in string.ascii_letters
        ]


def load_block_faces(library: str) -> dict[str, BlockFaces]:
    """The face map of a glyph library, by block name."""
    payload = json.loads((BLOCK_FACES_DIR / f"{library}.json").read_text(encoding="utf-8"))
    return {
        name: BlockFaces(name, record["usd"], record["faces"])
        for name, record in payload["blocks"].items()
    }


@dataclass(frozen=True)
class Family:
    """One task family, as `families.json` records it."""

    slug: str
    word: str
    collection: str
    library: str
    inventory: int
    feasibility: str
    """`feasible` or `infeasible` (see `FEASIBILITIES`)."""
    absent_letter: str | None
    """The target letter no block carries, in an infeasible family; None in a feasible one."""
    targets: tuple[str, str, str]
    """The designated block for each letter of the word, first letter first (the top of the
    goal tower), used only as the PDDL placeholder goal. Success does not depend on these: any
    blocks that spell the word count. In an infeasible family the absent letter's slot names the
    block that replaced it, so the goal is well formed and unsatisfiable."""
    blocks: dict[str, dict[str, Any]]
    """Block name to `{"kind": "letter" | "color", "type": pddl type, "letter_faces":
    {face: {"glyph", "up"}}}`."""
    on: dict[str, str]
    """Top block to the block it starts on."""
    on_table: tuple[str, ...]
    ori: dict[str, str]
    instances: tuple[str, ...]
    """Instance slugs, each one PDDL file and one package."""

    def letter_faces(self, name: str) -> list[LetterFace]:
        return [
            LetterFace(face, info["glyph"], info["up"])
            for face, info in self.blocks[name].get("letter_faces", {}).items()
        ]

    @property
    def feasible(self) -> bool:
        return self.feasibility == "feasible"

    def target_faces(self, name: str) -> list[LetterFace]:
        """The faces of block `name` carrying a letter of the word, in either case."""
        return [face for face in self.letter_faces(name) if face.glyph.upper() in self.word]

    def to_json(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "word": self.word,
            "collection": self.collection,
            "library": self.library,
            "inventory": self.inventory,
            "feasibility": self.feasibility,
            "absent_letter": self.absent_letter,
            "targets": list(self.targets),
            "blocks": self.blocks,
            "on": self.on,
            "on_table": list(self.on_table),
            "ori": self.ori,
            "instances": list(self.instances),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Family:
        return cls(
            slug=payload["slug"],
            word=payload["word"],
            collection=payload["collection"],
            library=payload["library"],
            inventory=int(payload["inventory"]),
            feasibility=payload["feasibility"],
            absent_letter=payload.get("absent_letter"),
            targets=tuple(payload["targets"]),
            blocks=payload["blocks"],
            on=payload["on"],
            on_table=tuple(payload["on_table"]),
            ori=payload["ori"],
            instances=tuple(payload["instances"]),
        )


@dataclass
class TaskFamilies:
    """The `families.json` of one task key."""

    task: str
    word: str
    collection: str
    library: str
    families: dict[str, Family] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return PROBLEMS_DIR / self.task / "families.json"

    def write(self, problems_dir: Path = PROBLEMS_DIR) -> Path:
        path = problems_dir / self.task / "families.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": self.task,
            "word": self.word,
            "collection": self.collection,
            "library": self.library,
            "specification": {
                "scene_blocks": SCENE_BLOCKS,
                "letter_blocks_range": list(LETTER_BLOCKS_RANGE),
                "stacks_range": list(STACKS_RANGE),
                "min_separation_m": MIN_SEPARATION_M,
                "block_size_range": list(BLOCK_SIZE_RANGE),
                "table_height_range": list(TABLE_HEIGHT_RANGE),
                "friction_range": list(FRICTION_RANGE),
            },
            "families": {slug: family.to_json() for slug, family in self.families.items()},
        }
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, task: str, problems_dir: Path = PROBLEMS_DIR) -> TaskFamilies:
        payload = json.loads((problems_dir / task / "families.json").read_text(encoding="utf-8"))
        return cls(
            task=payload["task"],
            word=payload["word"],
            collection=payload["collection"],
            library=payload["library"],
            families={
                slug: Family.from_json(record) for slug, record in payload["families"].items()
            },
        )


def task_key(word: str, collection: str) -> str:
    return f"spell_{word.lower()}_{collection}"


def letter_blocks_for(inventory: int, max_letter_blocks: int = LETTER_BLOCKS_RANGE[1]) -> int:
    """Letter blocks in inventory k of a word: `LETTER_BLOCKS_RANGE[0] + k`, so 6, 7 and 8 for
    k = 0, 1, 2 under the suite's range; `max_letter_blocks` raises the limit for task keys
    outside the suite (the example families run to ten, a scene with no color cube)."""
    count = LETTER_BLOCKS_RANGE[0] + inventory
    if count > max_letter_blocks:
        raise ValueError(f"inventory {inventory} would hold {count} letter blocks")
    return count


def family_slug(word: str, collection: str, inventory: int, feasibility: str) -> str:
    return f"{task_key(word, collection)}_{inventory}_{feasibility}"


def instance_slug(family: str, instance: int) -> str:
    return f"{family}_{instance}"


def parse_slug(slug: str) -> dict[str, str] | None:
    """The parts of a family or instance slug, or None for a slug of some other task."""
    match = SLUG_PATTERN.match(slug)
    return match.groupdict() if match else None


def family_of_instance(slug: str, problems_dir: Path = PROBLEMS_DIR) -> Family | None:
    """The family an instance slug belongs to, or None when the slug is not a Search & Spell
    instance or its task has no `families.json`."""
    parts = parse_slug(slug)
    if parts is None or parts["instance"] is None:
        return None
    path = problems_dir / parts["task"] / "families.json"
    if not path.is_file():
        return None
    families = TaskFamilies.read(parts["task"], problems_dir).families
    return families.get(slug.rsplit("_", 1)[0])


def task_keys() -> list[str]:
    """Every Search & Spell task key the specification defines."""
    return [task_key(word, collection) for word in WORDS for collection in COLLECTIONS]
