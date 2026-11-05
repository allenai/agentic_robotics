"""Generate the Search & Spell task families and instances as PDDL, to the developer's
specification ("Rewritten Evaluation Task Specifications", 2026-10-21, revised 2026-10-23: 6-8
letter blocks, hidden target letters only, feasible and infeasible families, both collections
matched within a word and inventory).

Needs no `tmp3` and no Isaac, so it runs on the host and the files it writes are the
host user's::

    agentic-robotics-spell-env                     # write data/problems/spell_*/
    agentic-robotics-spell-env --preview AUK unpatterned 0 infeasible
    agentic-robotics-spell-env --instances 1       # a smaller suite

**What is drawn, and from what seed.** Every draw is a `random.Random` seeded from the names
of the thing being drawn plus `--seed`, so one family can be regenerated alone and adding
instances leaves the existing ones as they are. In order:

1. *Structure* (per word and inventory k, shared by both collections): the letter-block count,
   `6 + k`; the color cubes, drawn without replacement from the training six to fill the scene to
   `SCENE_BLOCKS`; the number of two-block stacks, uniform over `STACKS_RANGE` and drawn once per
   word (inventory 0's draw, shared by every inventory of the word, so a word's inventories differ
   in letter blocks alone); and the stack layout as pairs of numbered block slots. Slots `0 .. n_letter-1` are letter blocks with the
   target blocks first, the rest are cubes, so the patterned and unpatterned families of one word
   and inventory stack the same slots and differ only in the blocks that fill them.
2. *Inventory* (per collection): patterned, the target blocks are the word's three letter blocks
   (each carries its letter in both cases) and the distractors are drawn from the other letters, so
   no letter of the word is on two blocks. Unpatterned, the target blocks are every block of the
   36-block set carrying a letter of the word in either case: the set puts each case of each
   letter on exactly one block, so the scene holds exactly one uppercase and one lowercase of
   each target letter, on as many blocks as the set happens to use (six for the specification's
   five words); the distractors are drawn from the blocks carrying no letter of the word. The
   designated target per letter, for the PDDL placeholder goal, is one of its carriers.
3. *Orientations* (per collection): a block carrying a target letter is drawn from the
   orientations under which none of its target-letter faces is on the front, or on top while the
   block has nothing on it; every other block from all 24.
4. *The infeasible family* is derived from the feasible one: the absent letter is the word's
   letter `k mod 3`, every block carrying it is replaced in its slot by a block carrying no letter
   of the word (a fresh draw from the same pool as the distractors), the replacement gets a
   uniformly random orientation, and everything else is kept, so the two families differ by the
   replaced block or blocks alone. One exception: a block carrying a lookalike of the absent
   letter (`LOOKALIKES`: a lowercase glyph that reads as the absent letter's lowercase under a
   half turn) is replaced too, and the replacements carry no lookalike, so the task cannot be
   answered by turning a decoy over. A family that already holds no lookalike is drawn exactly as
   if the rule did not exist. A lookalike on a block that also carries a letter of the word stays,
   since that block is a target carrier; the report names it.

**The PDDL goal is a placeholder** for codegen: the designated target blocks stacked in word
order, top to bottom, with no orientation literal; in an infeasible family the absent letter's
slot names its replacement block, so the goal is well formed and unsatisfiable. The package's own
classifier therefore reads "the designated blocks are stacked in order" (`success_strict` in the
episode summary), and the task's success is `leap.spelling.search_spell_success`, which accepts
any three blocks whose front faces spell the word.

**Instances.** The specification samples the continuous parameters with the training data's
generator and settings, which is Stage-1 codegen (`generate_task_from_pddl.py`) under
`run_datagen_pipeline_config_20261009_112203.json`: block size 3-5 cm, table height -5 to +5
cm, friction 0.5, `train_blocks_large`. Each instance is written as its own PDDL file, identical
to its siblings but for the problem name, so Stage-1 (seeded from the slug) samples each one
separately and each package carries its own scene profile; `search_spell_build.py` builds them,
rejects a sampled state that breaks the 7 cm rule, and keeps one state per instance.
"""

from __future__ import annotations

import argparse
import random
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_robotics.adapters.toyblocks import runner_bootstrap as boot

boot.prepare_import_path()

from agentic_robotics.adapters.toyblocks.block_orientations import (
    ORIENTATIONS,
    front_face,
    top_face,
)
from agentic_robotics.adapters.toyblocks.release.search_spell_families import (
    BLOCK_SIZE_RANGE,
    COLLECTIONS,
    COLOR_CUBES,
    FEASIBILITIES,
    FRICTION_RANGE,
    LETTER_BLOCKS_RANGE,
    LETTER_TYPE,
    PROBLEMS_DIR,
    SCENE_BLOCKS,
    STACKS_RANGE,
    TABLE_HEIGHT_RANGE,
    WORDS,
    BlockFaces,
    Family,
    LetterFace,
    TaskFamilies,
    family_slug,
    instance_slug,
    letter_blocks_for,
    load_block_faces,
    task_key,
)

TABLE = "workspace_table"

DEFAULT_INVENTORIES = 3
DEFAULT_INSTANCES = 2
"""The specification's counts: 3 inventories per word and collection (6, 7 and 8 letter blocks),
a feasible and an infeasible family per inventory, 2 instances per family: 60 families and 120
instances in all."""


def _rng(seed: int, *parts: Any) -> random.Random:
    return random.Random(f"{seed}:" + ":".join(str(part) for part in parts))


def _word_letter_faces(block: BlockFaces, word: str) -> list[LetterFace]:
    """The faces of a block carrying a letter of `word`, in either case."""
    return [face for face in block.letters() if face.glyph.upper() in word]


@dataclass(frozen=True)
class Structure:
    """What one word and inventory share across the two collections."""

    n_letter: int
    colors: tuple[str, ...]
    on_slots: dict[int, int]
    """Top slot to the slot it starts on."""


def _structure_draws(
    word: str, inventory: int, seed: int, max_letter_blocks: int
) -> tuple[int, tuple[str, ...], int, list[int]]:
    """One inventory's structure stream: letter-block count, cubes, a stack count, slot order."""
    rng = _rng(seed, word, inventory, "structure")
    n_letter = letter_blocks_for(inventory, max_letter_blocks)
    colors = tuple(rng.sample(sorted(COLOR_CUBES), SCENE_BLOCKS - n_letter))
    n_stacks = rng.randint(*STACKS_RANGE)
    order = rng.sample(range(SCENE_BLOCKS), SCENE_BLOCKS)
    return n_letter, colors, n_stacks, order


def draw_structure(
    word: str, inventory: int, seed: int, max_letter_blocks: int = LETTER_BLOCKS_RANGE[1]
) -> Structure:
    """The structure of one word and inventory.

    The stack count is the word's, the one its smallest inventory draws, so the inventories of a
    word differ in how many letter blocks there are and not in how many stacks must be taken
    apart; the cubes and the slot order stay the inventory's own draws.
    """
    n_letter, colors, _, order = _structure_draws(word, inventory, seed, max_letter_blocks)
    n_stacks = _structure_draws(word, 0, seed, max_letter_blocks)[2]
    on_slots = {order[2 * i]: order[2 * i + 1] for i in range(n_stacks)}
    return Structure(n_letter=n_letter, colors=colors, on_slots=on_slots)


def _carriers(faces: dict[str, BlockFaces], letter: str) -> list[str]:
    """The blocks carrying `letter` in either case, sorted by name."""
    return sorted(name for name, block in faces.items() if _has_letter(block, letter))


LOOKALIKES: dict[str, str] = {"n": "u", "u": "n", "p": "d", "d": "p", "b": "q", "q": "b"}
"""Lowercase glyphs that read as each other under a half turn. An infeasible family carries no
lookalike of its absent letter, so the task is impossible by eye and not only by the scorer's
rule (which admits no half turn for these glyphs)."""


def _carries_glyph(block: BlockFaces, glyph: str) -> bool:
    return any(face.glyph == glyph for face in block.letters())


def distractor_pool(word: str, collection: str, faces: dict[str, BlockFaces]) -> list[str]:
    """Every block of the collection carrying no letter of `word`, sorted by name."""
    if collection == "patterned":
        return [f"block_{c.lower()}" for c in sorted(set(string.ascii_uppercase) - set(word))]
    return sorted(name for name, block in faces.items() if not _word_letter_faces(block, word))


def draw_inventory(
    word: str,
    collection: str,
    inventory: int,
    structure: Structure,
    faces: dict[str, BlockFaces],
    seed: int,
) -> tuple[list[str], tuple[str, str, str]]:
    """Draw the letter blocks of one inventory, target blocks first.

    :return: Letter-block names in slot order, and the designated target block per letter of the
        word (first letter first)
    :raises ValueError: If the collection cannot hold the word's letters in `structure.n_letter`
        blocks, or a block carries two letters of the word
    """
    rng = _rng(seed, word, collection, inventory)
    if collection == "patterned":
        target_blocks = [f"block_{letter.lower()}" for letter in word]
        designated = tuple(target_blocks)
    else:
        carriers = {letter: _carriers(faces, letter) for letter in word}
        for letter, names in carriers.items():
            cases = sorted(
                face.glyph
                for name in names
                for face in faces[name].letters()
                if face.glyph.upper() == letter
            )
            if cases != sorted([letter, letter.lower()]):
                raise ValueError(f"{letter} is not on exactly one face per case: {cases}")
        target_blocks = sorted({name for names in carriers.values() for name in names})
        for name in target_blocks:
            letters = {face.glyph.upper() for face in _word_letter_faces(faces[name], word)}
            if len(letters) > 1:
                raise ValueError(f"{name} carries two letters of {word}: {sorted(letters)}")
        designated = tuple(rng.choice(carriers[letter]) for letter in word)
    if len(target_blocks) > structure.n_letter:
        raise ValueError(
            f"{word} ({collection}) needs {len(target_blocks)} target blocks, more than the "
            f"{structure.n_letter} letter blocks of inventory {inventory}"
        )
    pool = [name for name in distractor_pool(word, collection, faces) if name not in target_blocks]
    distractors = rng.sample(pool, structure.n_letter - len(target_blocks))
    return target_blocks + distractors, designated  # type: ignore[return-value]


def _has_letter(block: BlockFaces, letter: str) -> bool:
    return any(face.glyph.upper() == letter for face in block.letters())


def letter_visible(face: str, ori: str, clear: bool) -> bool:
    """The specification's visibility rule: on the front, or on top of a block with nothing on it."""
    return front_face(ori) == face or (clear and top_face(ori) == face)


def hiding_orientations(faces: list[LetterFace], clear: bool) -> list[str]:
    """Orientations under which none of `faces` is visible."""
    return [
        ori
        for ori in ORIENTATIONS
        if not any(letter_visible(face.face, ori, clear) for face in faces)
    ]


def _block_records(
    letter_names: list[str], structure: Structure, faces: dict[str, BlockFaces]
) -> dict[str, dict[str, Any]]:
    """Block records in slot order: the letter blocks, then the cubes."""
    blocks: dict[str, dict[str, Any]] = {}
    for name in letter_names:
        blocks[name] = {
            "kind": "letter",
            "type": LETTER_TYPE,
            "usd": faces[name].usd,
            "letter_faces": {
                face.face: {"glyph": face.glyph, "up": face.up} for face in faces[name].letters()
            },
        }
    for color in structure.colors:
        name, pddl_type = COLOR_CUBES[color]
        blocks[name] = {"kind": "color", "type": pddl_type, "letter_faces": {}}
    return blocks


def _hidden_orientation(rng: random.Random, word: str, record: dict[str, Any], clear: bool) -> str:
    """A uniformly random orientation hiding every face of `record` carrying a letter of `word`."""
    target_faces = [
        LetterFace(face, info["glyph"], info["up"])
        for face, info in record["letter_faces"].items()
        if info["glyph"].upper() in word
    ]
    choices = hiding_orientations(target_faces, clear) if target_faces else list(ORIENTATIONS)
    return rng.choice(choices)


def draw_feasible_family(
    word: str,
    collection: str,
    inventory: int,
    structure: Structure,
    faces: dict[str, BlockFaces],
    instances: int,
    seed: int,
) -> Family:
    """The feasible family of one word, collection and inventory."""
    letter_names, designated = draw_inventory(word, collection, inventory, structure, faces, seed)
    blocks = _block_records(letter_names, structure, faces)
    slots = list(blocks)
    on = {slots[top]: slots[bottom] for top, bottom in structure.on_slots.items()}
    covered = set(on.values())
    rng = _rng(seed, word, collection, inventory, "orientations")
    ori = {
        name: _hidden_orientation(rng, word, record, name not in covered)
        for name, record in blocks.items()
    }
    return Family(
        slug=family_slug(word, collection, inventory, "feasible"),
        word=word,
        collection=collection,
        library=COLLECTIONS[collection],
        inventory=inventory,
        feasibility="feasible",
        absent_letter=None,
        targets=designated,
        blocks=blocks,
        on=on,
        on_table=tuple(name for name in blocks if name not in on),
        ori=ori,
        instances=tuple(
            instance_slug(family_slug(word, collection, inventory, "feasible"), j)
            for j in range(instances)
        ),
    )


def derive_infeasible_family(
    feasible: Family, faces: dict[str, BlockFaces], instances: int, seed: int
) -> Family:
    """The infeasible counterpart of a feasible family: one target letter's block or blocks
    replaced in place by blocks carrying no letter of the word."""
    word, collection, k = feasible.word, feasible.collection, feasible.inventory
    absent = word[k % len(word)]
    rng = _rng(seed, word, collection, k, "infeasible")
    removed = [
        name
        for name in feasible.blocks
        if any(face.glyph.upper() == absent for face in feasible.letter_faces(name))
    ]
    if not removed:
        raise ValueError(f"{feasible.slug}: no block carries {absent}")
    used = set(feasible.blocks)
    pool = [name for name in distractor_pool(word, collection, faces) if name not in used]
    replacement = dict(zip(removed, rng.sample(pool, len(removed)), strict=True))
    lookalike = LOOKALIKES.get(absent.lower())
    if lookalike is not None and any(
        _carries_glyph(faces[name], lookalike)
        for name in [*feasible.blocks, *replacement.values()]
        if feasible.blocks.get(name, {"kind": "letter"})["kind"] == "letter"
    ):
        removed += [
            name
            for name, rec in feasible.blocks.items()
            if rec["kind"] == "letter"
            and name not in removed
            and _carries_glyph(faces[name], lookalike)
            and not _word_letter_faces(faces[name], word)
        ]
        pool = [name for name in pool if not _carries_glyph(faces[name], lookalike)]
        replacement = dict(zip(removed, rng.sample(pool, len(removed)), strict=True))

    def sub(name: str) -> str:
        return replacement.get(name, name)

    letter_names = [sub(name) for name, rec in feasible.blocks.items() if rec["kind"] == "letter"]
    cube_colors = {cube: color for color, (cube, _) in COLOR_CUBES.items()}
    structure = Structure(
        n_letter=len(letter_names),
        colors=tuple(
            cube_colors[name] for name, rec in feasible.blocks.items() if rec["kind"] == "color"
        ),
        on_slots={},
    )
    blocks = _block_records(letter_names, structure, faces)
    on = {sub(top): sub(bottom) for top, bottom in feasible.on.items()}
    ori = {sub(name): o for name, o in feasible.ori.items()}
    for name in replacement.values():
        ori[name] = rng.choice(list(ORIENTATIONS))
    slug = family_slug(word, collection, k, "infeasible")
    return Family(
        slug=slug,
        word=word,
        collection=collection,
        library=feasible.library,
        inventory=k,
        feasibility="infeasible",
        absent_letter=absent,
        targets=tuple(sub(name) for name in feasible.targets),  # type: ignore[arg-type]
        blocks={name: blocks[name] for name in blocks},
        on=on,
        on_table=tuple(name for name in blocks if name not in on),
        ori={name: ori[name] for name in blocks},
        instances=tuple(instance_slug(slug, j) for j in range(instances)),
    )


def visible_target_letters(family: Family) -> list[str]:
    """Which target letters are visible at the start, as `<glyph>@<block>`; for the report."""
    covered = set(family.on.values())
    return [
        f"{face.glyph}@{name}"
        for name in family.blocks
        for face in family.target_faces(name)
        if letter_visible(face.face, family.ori[name], name not in covered)
    ]


def render_pddl(family: Family, problem_name: str) -> str:
    """The PDDL problem text of one instance of a family, in the training problems' shape."""
    by_type: dict[str, list[str]] = {}
    for name, info in family.blocks.items():
        by_type.setdefault(info["type"], []).append(name)
    objects = "\n".join(f"    {' '.join(names)} - {kind}" for kind, names in by_type.items())

    init: list[str] = []
    for name in family.blocks:
        if name in family.on:
            init.append(f"(on {name} {family.on[name]})")
        else:
            init.append(f"(on_table {name} {TABLE})")
    covered = set(family.on.values())
    init += [f"(clear {name})" for name in family.blocks if name not in covered]
    init += [f"(ori_of {name} {family.ori[name]})" for name in family.blocks]
    init += ["(handempty robot)", "(gripper_at_init)"]

    top, middle, bottom = family.targets
    goal = [
        f"(on_table {bottom} {TABLE})",
        f"(on {middle} {bottom})",
        f"(on {top} {middle})",
        f"(clear {top})",
    ]
    init_text = "\n".join(f"        {literal}" for literal in init)
    goal_text = "\n".join(f"        {literal}" for literal in goal)
    return (
        f"(define (problem {problem_name})\n"
        f"    (:domain blocks-tamp)\n"
        f"    (:objects\n{objects}\n"
        f"    {TABLE} - table\n"
        f"    robot - gripper    )\n"
        f"    (:init\n{init_text}\n    )\n"
        f"    (:goal (and\n{goal_text}\n    ))\n\n"
        f"    ;block_size_range = {list(BLOCK_SIZE_RANGE)}\n"
        f"    ;table_height_range = {list(TABLE_HEIGHT_RANGE)}\n"
        f"    ;friction_range = {list(FRICTION_RANGE)}\n"
        f")\n"
    )


def generate_task(
    word: str,
    collection: str,
    *,
    inventories: int,
    instances: int,
    seed: int,
    faces: dict[str, BlockFaces],
    max_letter_blocks: int = LETTER_BLOCKS_RANGE[1],
    feasible_only: bool = False,
) -> TaskFamilies:
    """Every family of one task key, drawn but not written; `feasible_only` skips the derived
    infeasible families (the example tasks the agent experiments on are all feasible)."""
    task = TaskFamilies(
        task=task_key(word, collection),
        word=word,
        collection=collection,
        library=COLLECTIONS[collection],
    )
    for k in range(inventories):
        structure = draw_structure(word, k, seed, max_letter_blocks)
        feasible = draw_feasible_family(word, collection, k, structure, faces, instances, seed)
        task.families[feasible.slug] = feasible
        if not feasible_only:
            infeasible = derive_infeasible_family(feasible, faces, instances, seed)
            task.families[infeasible.slug] = infeasible
    return task


def check_family(family: Family, max_letter_blocks: int = LETTER_BLOCKS_RANGE[1]) -> None:
    """The specification's abstract constraints, checked on a drawn family.

    :raises AssertionError: On any violation
    """
    blocks = family.blocks
    assert len(blocks) == SCENE_BLOCKS
    n_letter = sum(1 for info in blocks.values() if info["kind"] == "letter")
    assert n_letter == letter_blocks_for(family.inventory, max_letter_blocks), n_letter
    assert STACKS_RANGE[0] <= len(family.on) <= STACKS_RANGE[1]
    bottoms = set(family.on.values())
    assert not (set(family.on) & bottoms), "a stack of three"
    assert len(bottoms) == len(family.on), "two blocks on one block"
    assert set(family.on) | set(family.on_table) == set(blocks)
    assert set(family.ori) == set(blocks)
    assert family.feasibility in FEASIBILITIES
    assert len(set(family.targets)) == 3

    def carriers(glyph: str) -> list[str]:
        return [n for n in blocks if any(f.glyph == glyph for f in family.letter_faces(n))]

    for letter, name in zip(family.word, family.targets, strict=True):
        if letter == family.absent_letter:
            assert not carriers(letter) and not carriers(letter.lower()), f"{letter} present"
            lookalike = LOOKALIKES.get(letter.lower())
            for carrier in carriers(lookalike) if lookalike else []:
                # A lookalike on a block that also carries a letter of the word cannot be
                # replaced, so it stays; the generator names it in its report.
                assert any(f.glyph.upper() in family.word for f in family.letter_faces(carrier)), (
                    f"{carrier} carries {lookalike}, which reads as {letter}"
                )
            continue
        assert any(f.glyph.upper() == letter for f in family.letter_faces(name)), (
            f"{name} does not carry {letter}"
        )
        assert len(carriers(letter)) == 1 and len(carriers(letter.lower())) == 1, (
            f"{letter}: {carriers(letter)} / {carriers(letter.lower())}"
        )
    assert (family.absent_letter is None) == family.feasible
    assert not visible_target_letters(family), visible_target_letters(family)


def write_tasks(tasks: list[TaskFamilies], problems_dir: Path) -> None:
    for task in tasks:
        for family in task.families.values():
            for slug in family.instances:
                path = problems_dir / task.task / f"{slug}.pddl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(render_pddl(family, slug), encoding="utf-8")
        path = task.write(problems_dir)
        print(f"{task.task}: {len(task.families)} families -> {path.parent}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--words", nargs="+", default=list(WORDS))
    parser.add_argument(
        "--collections", nargs="+", default=list(COLLECTIONS), choices=list(COLLECTIONS)
    )
    parser.add_argument("--inventories", type=int, default=DEFAULT_INVENTORIES)
    parser.add_argument("--instances", type=int, default=DEFAULT_INSTANCES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=PROBLEMS_DIR)
    parser.add_argument(
        "--max-letter-blocks",
        type=int,
        default=LETTER_BLOCKS_RANGE[1],
        help="Letter blocks allowed in the largest inventory (the suite's range tops out at 8; "
        "a scene of 10 letter blocks has no color cube)",
    )
    parser.add_argument("--feasible-only", action="store_true", help="Draw no infeasible families")
    parser.add_argument(
        "--preview",
        nargs=4,
        metavar=("WORD", "COLLECTION", "INVENTORY", "FEASIBILITY"),
        default=None,
        help="Print one family's PDDL and its blocks instead of writing",
    )
    args = parser.parse_args()
    faces = {library: load_block_faces(library) for library in COLLECTIONS.values()}

    if args.preview:
        word, collection, k, feasibility = args.preview
        task = generate_task(
            word.upper(),
            collection,
            inventories=int(k) + 1,
            instances=args.instances,
            seed=args.seed,
            faces=faces[COLLECTIONS[collection]],
            max_letter_blocks=args.max_letter_blocks,
            feasible_only=args.feasible_only,
        )
        family = task.families[family_slug(word, collection, int(k), feasibility)]
        check_family(family, args.max_letter_blocks)
        print(render_pddl(family, family.instances[0]))
        print("targets:", family.targets, "absent:", family.absent_letter)
        for name, record in family.blocks.items():
            glyphs = ",".join(f["glyph"] for f in record["letter_faces"].values()) or "-"
            print(f"  {name:14s} {family.ori[name]:4s} letters {glyphs}")
        return 0

    tasks = [
        generate_task(
            word.upper(),
            collection,
            inventories=args.inventories,
            instances=args.instances,
            seed=args.seed,
            faces=faces[COLLECTIONS[collection]],
            max_letter_blocks=args.max_letter_blocks,
            feasible_only=args.feasible_only,
        )
        for word in args.words
        for collection in args.collections
    ]
    for task in tasks:
        for family in task.families.values():
            check_family(family, args.max_letter_blocks)
            lookalike = LOOKALIKES.get((family.absent_letter or "").lower())
            kept = [
                name
                for name in family.blocks
                if lookalike and any(f.glyph == lookalike for f in family.letter_faces(name))
            ]
            if kept:
                print(
                    f"{family.slug}: {lookalike} (reads as {family.absent_letter}) stays on {kept}"
                )
    write_tasks(tasks, args.out_dir)
    n_families = sum(len(task.families) for task in tasks)
    n_instances = sum(len(f.instances) for task in tasks for f in task.families.values())
    by_kind = {
        kind: sum(1 for task in tasks for f in task.families.values() if f.feasibility == kind)
        for kind in FEASIBILITIES
    }
    print(f"{len(tasks)} tasks, {n_families} families ({by_kind}), {n_instances} instances.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
