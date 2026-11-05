"""Which glyph is on which body face of each block of a glyph library, and which way it reads.

A probe in the sense of CLAUDE.md: it reads USD files, runs no agent and drives no simulator.
It needs `pxr` (the Isaac venv in the container) and writes `data/block_faces/<library>.json`,
which the Search & Spell generator (`search_spell_env.py`) and scorer (`leap/spelling.py`)
read, so neither of them needs USD at run time::

    # In the container, where pxr lives; write to a directory the host user can copy from
    python scripts/decode_block_faces.py \\
        --library eval_blocks_36_0 --out data/block_faces
    python scripts/decode_block_faces.py \\
        --library train_blocks_large --out data/block_faces

Method. Each block USD carries one decal mesh per face, a quad whose vertices have texture coordinates
into a 16 x 16 glyph atlas. The quad's centroid names the body face (the axis of largest
magnitude), the mean texture coordinate names the atlas cell, and the cell names the glyph
through the library's `dataset_description.json` legend. For the reading direction, a least
squares fit of vertex position against texture coordinate gives the 3D direction along which
`v` increases, and since the atlas is indexed with `v` flipped (row 0 at `v = 1`), that is the
direction the glyph's top points in, in the block's body frame. Each face record carries
`glyph`, `type` (`character` or `symbol`), `up` (the body axis the glyph's top points along)
and `right`.

Known answer. `train_blocks_large` has no per-cell legend; its 26 letter files (the two digit
files, 2 and 9, are laid out differently and are skipped) each use a 3 x 2 block of atlas cells, and rendering shows that
the uppercase letter is on body +y and the lowercase on -y, both reading upright when body +z is
up (orientations `o2` and `o4`). Decoding that library checks the method: every letter file must
put its +y and -y faces at the cells the pattern predicts, with `up` = +z on both, or the
script fails. The output for that library records the letters as `L` on +y and `l` on -y and
the other four faces as `picture`.

Handedness is checked on every face: `right` must equal `up` x (outward normal), which is what a
viewer looking at the face sees, so a mirrored texture mapping fails loudly.
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path
from typing import Any

import numpy as np

from agentic_robotics.adapters.toyblocks.runner_bootstrap import toyblocks_root

ASSETS_DIR = toyblocks_root() / "assets/objects"
"""Where the ToyBlocks checkout keeps the glyph libraries (`toyblocks.OBJECT_ASSET_DIR`)."""

ATLAS_GRID = 16
"""Cells per side of the glyph atlas texture."""

AXES: dict[str, np.ndarray] = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


def _under_toyblocks_home(path: Path) -> str:
    """``path`` relative to the ``.toyblocks`` directory it sits under, so the record carries no
    machine's absolute path; the whole path if it is elsewhere."""
    parts = path.parts
    if ".toyblocks" in parts:
        return str(Path(*parts[parts.index(".toyblocks") + 1 :]))
    return str(path)


def _axis_name(vector: np.ndarray) -> str:
    """The signed body axis a vector lies closest to, e.g. `-y`."""
    index = int(np.argmax(np.abs(vector)))
    return ("+" if vector[index] > 0 else "-") + "xyz"[index]


def decode_usd(path: Path) -> dict[str, dict[str, Any]]:
    """Decode one block USD: body face -> atlas cell, glyph up axis, glyph right axis.

    :param path: A block `.usda`
    :return: Face name (`+x` .. `-z`) to `{"cell": (col, row), "up": axis, "right": axis}`
    :raises ValueError: If a face's texture mapping is mirrored
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    faces: dict[str, dict[str, Any]] = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh" or "decal" not in prim.GetName().lower():
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get())
        quads = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 4)
        st = np.array(UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get()).reshape(-1, 4, 2)
        for quad, uv in zip(quads, st, strict=True):
            corners = points[quad]
            center = corners.mean(0)
            face = _axis_name(center)
            # Position as an affine function of (u, v): the rows of the fit are the 3D
            # directions of increasing u and increasing v.
            fit, *_ = np.linalg.lstsq(uv - uv.mean(0), corners - center, rcond=None)
            right, up = _axis_name(fit[0]), _axis_name(fit[1])
            expected_right = _axis_name(np.cross(AXES[up], AXES[face]))
            if right != expected_right:
                raise ValueError(
                    f"{path.name} face {face}: right is {right}, but up {up} x normal {face} "
                    f"gives {expected_right}; the texture mapping is mirrored."
                )
            mean_uv = uv.mean(0)
            cell = (int(mean_uv[0] * ATLAS_GRID), int((1.0 - mean_uv[1]) * ATLAS_GRID))
            faces[face] = {"cell": cell, "up": up, "right": right}
    if sorted(faces) != sorted(AXES):
        raise ValueError(f"{path.name}: decoded faces {sorted(faces)}, expected all six.")
    return faces


def decode_eval_library(library_dir: Path) -> dict[str, Any]:
    """Decode a library whose `dataset_description.json` carries a per-cell glyph legend."""
    description = json.loads((library_dir / "dataset_description.json").read_text())
    legend = {tuple(entry["cell"]): entry for entry in description["glyph_legend"]}
    blocks: dict[str, dict[str, Any]] = {}
    for usd in sorted(library_dir.glob("block_*.usda")):
        faces = decode_usd(usd)
        record: dict[str, Any] = {"usd": usd.name, "faces": {}}
        for face, info in faces.items():
            entry = legend.get(info["cell"])
            if entry is None:
                raise ValueError(
                    f"{usd.name} face {face}: cell {info['cell']} is not in the legend."
                )
            record["faces"][face] = {
                "glyph": entry["glyph"],
                "type": entry["type"],
                "up": info["up"],
                "right": info["right"],
            }
        blocks[usd.stem] = record
    uses = [face["glyph"] for block in blocks.values() for face in block["faces"].values()]
    if len(uses) != len(set(uses)):
        raise ValueError("A glyph appears on more than one face of the library.")
    return {
        "library": library_dir.parent.name,
        "source": _under_toyblocks_home(library_dir),
        "blocks": blocks,
    }


def decode_large_library(library_dir: Path) -> dict[str, Any]:
    """Decode `train_blocks_large` against the known layout (the module docstring)."""
    blocks: dict[str, dict[str, Any]] = {}
    for usd in sorted(library_dir.glob("*.usda")):
        glyph = usd.stem
        if glyph not in string.ascii_uppercase:
            # The two digit files (2, 9) lay their cells out differently and carry no legend,
            # so which of their faces is the digit is not decodable here; the Search & Spell
            # patterned inventories use letters only (`search_spell_env.py`).
            continue
        faces = decode_usd(usd)
        columns = sorted({info["cell"][0] for info in faces.values()})
        rows = sorted({info["cell"][1] for info in faces.values()})
        if (
            len(columns) != 3
            or len(rows) != 2
            or columns[2] - columns[0] != 2
            or rows[1] - rows[0] != 1
        ):
            raise ValueError(f"{usd.name}: cells {faces} do not form the 3 x 2 glyph block.")
        base, row = columns[0], rows[0]
        if faces["+y"]["cell"] != (base + 1, row + 1) or faces["-y"]["cell"] != (base, row):
            raise ValueError(f"{usd.name}: +y/-y are not at the cells the layout predicts.")
        if faces["+y"]["up"] != "+z" or faces["-y"]["up"] != "+z":
            raise ValueError(f"{usd.name}: a letter face does not read upright with +z up.")
        record: dict[str, Any] = {"usd": usd.name, "faces": {}}
        for face, info in faces.items():
            if face == "+y":
                face_glyph, kind = glyph, "character"
            elif face == "-y":
                face_glyph, kind = glyph.lower(), "character"
            else:
                face_glyph, kind = "picture", "symbol"
            record["faces"][face] = {
                "glyph": face_glyph,
                "type": kind,
                "up": info["up"],
                "right": info["right"],
            }
        blocks[f"block_{glyph.lower()}"] = record
    return {
        "library": library_dir.parent.name,
        "source": _under_toyblocks_home(library_dir),
        "blocks": blocks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--library", required=True, choices=("eval_blocks_36_0", "train_blocks_large")
    )
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    parser.add_argument("--out", type=Path, required=True, help="Directory for <library>.json")
    args = parser.parse_args()
    library_dir = (args.assets_dir / args.library / "usd").resolve()
    if args.library == "eval_blocks_36_0":
        payload = decode_eval_library(library_dir)
    else:
        payload = decode_large_library(library_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.library}.json"
    out.write_text(json.dumps(payload, indent=1) + "\n")
    letters = sum(
        1
        for block in payload["blocks"].values()
        for face in block["faces"].values()
        if face["type"] == "character" and face["glyph"] in string.ascii_letters
    )
    print(f"{out}: {len(payload['blocks'])} blocks, {letters} letter faces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
