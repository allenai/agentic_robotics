"""Define utility functions to generate randomized toy blocks."""

import random
from collections.abc import Callable
from string import ascii_uppercase

TargetStringGenerator = Callable[[], str]


def generate_block_string(num_symbols: int = 0, include_faces: str = "") -> str:
    """Generate a six-character string representing letters on a randomized block's faces.

    :param num_symbols: Number of block faces with non-letter symbols (default: 0)
    :param include_faces: Letters included as faces of the block (defaults to none)
    :return: Six-character string of word characters (alphanumeric + underscores for symbol faces)
    """
    if not (0 <= num_symbols <= 6):
        raise ValueError(f"Invalid number of symbols to include on block faces: {num_symbols}.")

    if not include_faces.isalpha():
        raise ValueError(f"String of included letters must be alphabetic: '{include_faces}'.")

    free_faces = 6 - num_symbols - len(include_faces)

    if free_faces < 0:
        raise ValueError(f"Cannot create block using {num_symbols} symbols and '{include_faces}'.")

    letters = random.choices(ascii_uppercase, k=free_faces)
    symbols = ["_"] * num_symbols
    block_faces = letters + symbols + list(include_faces)
    random.shuffle(block_faces)

    return "".join(block_faces)


def generate_block_strings(get_string: TargetStringGenerator, num_blocks: int) -> list[str]:
    """Generate a list of block strings in which a sampled target string is admissible.

    :param get_string: Callable used to sample a target string
    :param num_blocks: Number of blocks in the generated scene specification
    :return: List of strings, each representing the glyphs on a block's six faces
    """
    target_string = get_string()
    num_distractors = num_blocks - len(target_string)

    if num_distractors < 0:
        raise ValueError(f"Cannot spell '{target_string}' using {num_blocks} blocks.")

    target_blocks = [generate_block_string(include_faces=char) for char in target_string]
    distractors = [generate_block_string() for _ in range(num_distractors)]

    all_blocks = target_blocks + distractors
    random.shuffle(all_blocks)

    return all_blocks
