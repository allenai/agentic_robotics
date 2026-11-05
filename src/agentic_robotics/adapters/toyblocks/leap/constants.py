"""Define constants used to construct agents for the LEAP experiments."""

# Step caps for the ManiFlow skills. The policy ends each call itself (its play-mode output), so
# the cap is a safety net: 105% of the longest single attempt in the training data
# (`oct06_stack3_1000x32`, segmented per attempt, 2026-10-20), rounded to the nearest integer. The
# agent cannot set them; the developer's decision of 2026-10-23.

PICK_STEPS = 237
"""Step cap for pick: the longest training attempt was 226."""

PLACE_STEPS = 200
"""Step cap for place on table: the longest training attempt was 190."""

STACK_STEPS = 204
"""Step cap for stack (place onto a block): the longest training attempt was 194."""

ROTATE_STEPS = 518
"""Step cap for rotate: the longest training attempt was 493 (clockwise)."""

FLIP_STEPS = 677
"""Step cap for flip, all four directions: the longest training attempt was 645."""
