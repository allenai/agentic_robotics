"""Refuse to drive a ManiFlow policy against blocks it was never trained on.

A policy sees pixels of a block and emits joint targets; nothing in the observation says which
USD spawned it. So a task package built against one glyph asset library and a checkpoint trained
against another produce a run that boots and executes every skill on blocks the policy has never
seen. The libraries differ in appearance (glyph style and color, picture faces, wood), not in size:
every spawned block takes its edge length from the task's `block_sizes`, which overrides the
library's `usd_scale` (`blocksuite.utils.rigid_objects.create_toy_block`).

The library is chosen at codegen time by `--usd_library`, which defaults to `train_blocks`, and
nothing else downstream compares the two. Both sides are on disk and neither needs Isaac:

- what a run will spawn: the task package's `env.py`, which carries its block table as a literal;
- what the policy trained on: the checkpoint's `config.yaml` names a `data_root_dir`, that
  dataset's `meta/provenance.json` names the `ll_env_directory` that recorded it, and that
  directory holds an `env.py` of the same shape.

Reading the dataset out of the checkpoint rather than out of the run configuration's
`maniflow.dataset_dirs` is deliberate: those are hand-set, so a `checkpoint_dir` and a dataset
entry pointing at unrelated things would compare the environment against itself and pass.

Two generations of codegen are in the tree and both are parsed here. The current one emits
`block_specs`, a dict of only the blocks the scene spawns. The older one (`wasp_tower`, `find_r`)
emits `block_names` beside a `usd_overrides` table holding every override ToyBlocks knows,
most of which that scene never spawns -- so the older form has to be intersected with
`block_names` or it reports assets no run would load.

The checkpoint side can also be read from a record under `data/checkpoint_block_assets/`,
written once by
`python -m agentic_robotics.adapters.toyblocks.block_asset_check record <checkpoint dir>` while
the chain above still resolves. The datasets tree is not ours and gets cleaned (the unmerged
`sep3_ai2_reorient_fixed` environments behind the served checkpoint are gone, so its chain
resolves nowhere but the workstation's stand-in), so the record is the durable copy and is
consulted first.

Stdlib only, so this runs in the light venv before `agentic-robotics` re-execs into the heavy one.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("agentic_robotics.examples.block_asset_check")

BlockAssets = dict[str, tuple[str, tuple[float, ...]]]
"""Block name to the glyph library it spawns from and the uniform USD scale applied to it."""


class BlockAssetMismatch(RuntimeError):
    """Raised when a task package's blocks are not the ones a checkpoint trained on."""


class BlockAssetsUnknown(RuntimeError):
    """Raised when either side cannot be read, so a match cannot be confirmed either way."""


def _literal_after(source: str, name: str) -> object | None:
    """Evaluate the literal assigned to `name` in generated code.

    Generated `env.py` bakes its tables in as one-line literals, so the assignment can be found by
    name and handed to `ast.literal_eval` without importing the module -- which would import
    ToyBlocks, and through it Isaac.

    :param source: Text of a generated `env.py`
    :param name: Variable assigned a list or dict literal
    :return: The parsed literal, or None when absent or not parseable
    """
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*([\[{{])", source, re.MULTILINE)
    if match is None:
        return None
    opening = match.group(1)
    closing = {"[": "]", "{": "}"}[opening]
    depth = 0
    start = match.end() - 1
    for index in range(start, len(source)):
        char = source[index]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                try:
                    return ast.literal_eval(source[start : index + 1])
                except (SyntaxError, ValueError):
                    return None
    return None


def _library_of(usd_path: str) -> str:
    """The glyph asset library a USD sub-path spawns from, e.g. `train_blocks_large`."""
    return usd_path.split("/", 1)[0]


def spawned_block_assets(env_py: Path) -> BlockAssets:
    """Read which glyph assets a generated env package spawns.

    :param env_py: Path to a generated `env.py`
    :return: Block name to (library, scale), for USD blocks only; procedural cuboids carry no
        asset and are left out
    :raises BlockAssetsUnknown: If the file cannot be read or carries neither table
    """
    try:
        source = env_py.read_text()
    except OSError as exc:
        raise BlockAssetsUnknown(f"Cannot read {env_py}: {exc}") from exc

    assets: BlockAssets = {}

    specs = _literal_after(source, "block_specs")
    if isinstance(specs, dict):
        for name, spec in specs.items():
            if isinstance(spec, dict) and spec.get("type") == "usd":
                assets[name] = (_library_of(spec["usd_path"]), tuple(spec["usd_scale"]))
        return assets

    # Older codegen: the override table lists every block ToyBlocks knows, so only the names
    # this scene actually spawns count.
    names = _literal_after(source, "block_names")
    overrides = _literal_after(source, "usd_overrides")
    if isinstance(names, list) and isinstance(overrides, dict):
        for name in names:
            override = overrides.get(name)
            if override is not None:
                assets[name] = (_library_of(override[0]), tuple(override[1]))
        return assets

    raise BlockAssetsUnknown(f"{env_py} carries neither `block_specs` nor `usd_overrides`.")


def training_env_pys(checkpoint_dir: Path) -> list[Path]:
    """Find the `env.py` of every environment a checkpoint's training data was recorded in.

    A checkpoint trained on several datasets names them comma-separated in one `data_root_dir`
    (the 2026-09-24 reorient checkpoint names three), and each is its own recording run.

    :param checkpoint_dir: A ManiFlow checkpoint directory
    :return: Path to each recording environment's generated `env.py`, in the config's order
    :raises BlockAssetsUnknown: If any link in checkpoint -> dataset -> env directory is missing
    """
    config_yaml = checkpoint_dir / "config.yaml"
    if not config_yaml.is_file():
        raise BlockAssetsUnknown(
            f"{checkpoint_dir.name} has no config.yaml, so it does not say which dataset it "
            f"trained on."
        )

    # Read the one line rather than the YAML: this runs in the venv that has no yaml.
    match = re.search(r"^\s*data_root_dir:\s*(\S+)", config_yaml.read_text(), re.MULTILINE)
    if match is None:
        raise BlockAssetsUnknown(f"{config_yaml} names no `data_root_dir`.")
    return [_dataset_env_py(Path(root)) for root in match.group(1).split(",") if root]


def training_env_py(checkpoint_dir: Path) -> Path:
    """The first of :func:`training_env_pys`, for callers that want one environment."""
    return training_env_pys(checkpoint_dir)[0]


def _dataset_env_py(dataset_dir: Path) -> Path:
    """The `env.py` of the environment that recorded one dataset, via its provenance file."""
    provenance = dataset_dir / "meta" / "provenance.json"
    if not provenance.is_file():
        raise BlockAssetsUnknown(f"{dataset_dir} has no meta/provenance.json.")

    # A merged dataset carries thousands of sources, all recorded by one datagen run against one
    # asset library; the first one answers the question the rest would repeat.
    try:
        source = json.loads(provenance.read_text())["sources"][0]
    except (json.JSONDecodeError, KeyError, IndexError, OSError) as exc:
        raise BlockAssetsUnknown(f"Cannot read a source from {provenance}: {exc}") from exc

    env_py = _source_env_py(source, provenance)
    if not env_py.is_file():
        raise BlockAssetsUnknown(f"{provenance} leads to {env_py}, which does not exist.")
    return env_py


def _source_env_py(source: dict, provenance: Path) -> Path:
    """The `env.py` one merged source was recorded in.

    A dataset recorded by datagen names it as `ll_env_directory` in its recording provenance. A
    re-rendered dataset (the `_render_N1` and `_allrand_N1` datasets behind the 2026-09-24
    checkpoint, and the re-partitioned `sep3_reorient_PARTIAL` one) leaves that empty and instead
    records `env_file` in the `meta/render_params.json` of the unmerged source dataset.

    :param source: One entry of the merged provenance's `sources`
    :param provenance: The merged provenance file, for error messages
    :return: Path to the recording environment's generated `env.py`
    :raises BlockAssetsUnknown: If neither record names an environment
    """
    recorded = source.get("recording_provenance_sources") or {}
    if "ll_env_directory" in recorded:
        return Path(recorded["ll_env_directory"]) / "env.py"

    render_params = Path(source.get("source_dataset", "")) / "meta" / "render_params.json"
    if not render_params.is_file():
        raise BlockAssetsUnknown(
            f"{provenance}'s first source names no `ll_env_directory` and has no "
            f"{render_params} to read an `env_file` from."
        )
    try:
        env_file = json.loads(render_params.read_text())["env_file"]
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        raise BlockAssetsUnknown(f"Cannot read `env_file` from {render_params}: {exc}") from exc
    return Path(env_file)


def _describe(assets: BlockAssets) -> str:
    """One line per distinct (library, scale), with the blocks that use it."""
    grouped: dict[tuple[str, tuple[float, ...]], list[str]] = {}
    for name, key in sorted(assets.items()):
        grouped.setdefault(key, []).append(name)
    return "\n".join(
        f"      {library} scale={scale} -> {', '.join(names)}"
        for (library, scale), names in sorted(grouped.items())
    )


RECORDED_ASSETS_DIR = Path(__file__).resolve().parents[4] / "data" / "checkpoint_block_assets"
"""Where :func:`record_trained_block_assets` keeps a checkpoint's block assets, by checkpoint
name."""


def recorded_assets_path(checkpoint_dir: Path) -> Path:
    return RECORDED_ASSETS_DIR / f"{checkpoint_dir.name}.json"


def recorded_block_assets(checkpoint_dir: Path) -> BlockAssets | None:
    """The record of a checkpoint's block assets, or None when none has been written."""
    path = recorded_assets_path(checkpoint_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise BlockAssetsUnknown(f"Cannot read {path}: {exc}") from exc
    return {name: (library, tuple(scale)) for name, (library, scale) in payload["assets"].items()}


def _under_datasets_root(path: Path) -> str:
    """``path`` relative to the datasets root it sits under, so the record carries no machine's
    absolute path: the part after the last ``datasets`` directory, or the whole path if there is
    none."""
    parts = path.parts
    if "datasets" in parts:
        return str(Path(*parts[len(parts) - parts[::-1].index("datasets") :]))
    return str(path)


def record_trained_block_assets(checkpoint_dir: Path) -> Path:
    """Write the checkpoint's block assets, read through the dataset chain, as a record.

    :param checkpoint_dir: Checkpoint being served
    :return: The record's path
    :raises BlockAssetsUnknown: If the chain checkpoint -> dataset -> environment breaks
    """
    assets: BlockAssets = {}
    sources = []
    for env in training_env_pys(checkpoint_dir):
        assets.update(spawned_block_assets(env))
        sources.append(_under_datasets_root(env))
    path = recorded_assets_path(checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint_dir.name,
                "training_env_pys": sources,
                "assets": {
                    name: [library, list(scale)]
                    for name, (library, scale) in sorted(assets.items())
                },
            },
            indent=1,
        )
        + "\n"
    )
    return path


_TRAINED_ASSETS: dict[Path, BlockAssets] = {}
"""Per-process cache of :func:`trained_block_assets`; a menu asks once per variant."""


def trained_block_assets(checkpoint_dir: Path) -> BlockAssets:
    """The blocks a checkpoint trained on, across every dataset it names.

    :param checkpoint_dir: Checkpoint being served
    :return: Block name to (library, scale), unioned over the training environments
    :raises BlockAssetsUnknown: If the chain checkpoint -> dataset -> environment breaks
    """
    cached = _TRAINED_ASSETS.get(checkpoint_dir)
    if cached is None:
        recorded = recorded_block_assets(checkpoint_dir)
        if recorded is not None:
            logger.info(
                "Block assets of %s read from %s.",
                checkpoint_dir.name,
                recorded_assets_path(checkpoint_dir),
            )
            cached = recorded
        else:
            cached = {}
            for env in training_env_pys(checkpoint_dir):
                cached.update(spawned_block_assets(env))
        _TRAINED_ASSETS[checkpoint_dir] = cached
    return cached


def _kinds_text(kinds: set[tuple[str, tuple[float, ...]]]) -> str:
    """`library at scale s` for each distinct kind, comma-separated."""
    return ", ".join(f"{library} at scale {scale[0]:g}" for library, scale in sorted(kinds))


def block_mismatch_reason(env_py: Path, checkpoint_dir: Path) -> str | None:
    """One line saying why a package's blocks do not fit the checkpoint, or None when they do.

    For a menu that greys a task out; :func:`assert_blocks_match` carries the full table.

    :param env_py: Generated `env.py` of a package
    :param checkpoint_dir: Checkpoint being served
    :return: The reason, or None when every block the package spawns was trained on
    :raises BlockAssetsUnknown: If either side cannot be read
    """
    deployed = spawned_block_assets(env_py)
    if not deployed:
        return None
    trained = trained_block_assets(checkpoint_dir)
    if not trained:
        return f"the environment behind {checkpoint_dir.name} spawns no glyph blocks to match"
    extra = set(deployed.values()) - set(trained.values())
    if not extra:
        return None
    return (
        f"blocks are {_kinds_text(extra)}; the policy trained on "
        f"{_kinds_text(set(trained.values()))}"
    )


def trained_libraries(checkpoint_dir: Path) -> set[str]:
    """The glyph libraries a checkpoint's training blocks spawn from."""
    return {library for library, _ in trained_block_assets(checkpoint_dir).values()}


def assert_blocks_from_library(env_py: Path, library: str) -> None:
    """Refuse a package whose glyph blocks do not all spawn from the library its task declares.

    For a task whose blocks are deliberately not the checkpoint's (the Search & Spell
    unpatterned collection, `eval_blocks_36_0`): the comparison that still catches
    a build against the wrong library is against the declaration, not the checkpoint.

    :param env_py: Generated `env.py` of the package about to be loaded
    :param library: The library the task declares
    :raises BlockAssetMismatch: If any USD block spawns from another library
    :raises BlockAssetsUnknown: If the package cannot be read
    """
    deployed = spawned_block_assets(env_py)
    wrong = {name: lib for name, (lib, _) in deployed.items() if lib != library}
    if wrong:
        raise BlockAssetMismatch(
            f"{env_py.parent.name} declares {library} but spawns "
            + ", ".join(f"{name} from {lib}" for name, lib in sorted(wrong.items()))
        )


def assert_blocks_match(env_py: Path, checkpoint_dir: Path) -> None:
    """Refuse a run whose blocks are not the ones the checkpoint trained on.

    :param env_py: Generated `env.py` of the package about to be loaded
    :param checkpoint_dir: Checkpoint being served
    :raises BlockAssetMismatch: If the libraries or scales differ
    :raises BlockAssetsUnknown: If either side cannot be read
    """
    deployed = spawned_block_assets(env_py)
    if not deployed:
        logger.info("%s spawns no glyph blocks; nothing to match.", env_py.parent.name)
        return

    trained = trained_block_assets(checkpoint_dir)
    if not trained:
        raise BlockAssetsUnknown(
            f"The environment behind {checkpoint_dir.name} spawns no glyph blocks, so there is "
            f"nothing to match {env_py.parent.name} against."
        )

    deployed_kinds = set(deployed.values())
    trained_kinds = set(trained.values())
    if deployed_kinds <= trained_kinds:
        return

    raise BlockAssetMismatch(
        f"Task package '{env_py.parent.name}' spawns blocks the policy never trained on.\n"
        f"    the run would spawn:\n{_describe(deployed)}\n"
        f"    the policy trained on:\n{_describe(trained)}\n"
        f"    Regenerate the task with the library the checkpoint used -- pass "
        f"`--usd_library <library>` to scripts/generate_task_from_pddl.py -- or serve a "
        f"checkpoint trained against these blocks."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record a checkpoint's block assets under data/.")
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="Read the dataset chain and write the record")
    record.add_argument("checkpoint_dir", type=Path)
    arguments = parser.parse_args()
    written = record_trained_block_assets(arguments.checkpoint_dir)
    print(written)
    print(_describe(recorded_block_assets(arguments.checkpoint_dir) or {}))
