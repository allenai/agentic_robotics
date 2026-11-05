"""Define handwritten "ground-truth" docstrings for an agent with the ManiFlow motor skills.

These descriptions are based on observations of the LEA ManiFlow checkpoint.
"""

DOCSTRINGS: dict[str, str] = {
    "pick": """Pick up the block at the given pixel in the current image.

    Select the target block's center pixel when calling `pick`.

    Failures are more common at the table edges and near clusters of blocks.

    Precondition: The robot's gripper must be empty before it can `pick` a block.

    Effect: The robot's gripper closes on the block and lifts it 2-5 cm. When `pick`
        fails, the gripper closes on nothing, typically beside or above the target block.

    :param pixel: Raw (u,v) pixel coordinate of the block to pick
    :return: Updated image observation after the skill has finished
    """,
    #
    #
    "stack": """Stack the held block onto the block at the given pixel in the current image.

    Select the base block's center pixel when calling `stack`.

    Precondition: The robot must already be holding a block.

    Effect: The stacked block will be centered within 1.5 cm on the target block about 3/4
        of the time. Sometimes, `stack` knocks the base block over, or ends without releasing
        the held block onto the base block, or drops the held block next to the base block.

    After a `stack`, check if the block is still in the gripper before the next `pick`.

    :param pixel: Raw (u,v) pixel coordinate of the block to be stacked on
    :return: Updated image observation after the skill has finished
    """,
    #
    #
    "place_on_table": """Place the held block onto the table, at the given pixel in the current image.

    Select the pixel where the held block should come to rest when calling `place_on_table`.

    Precondition: The robot must already be holding a block.

    Effect: The held block will be placed on the table, within 3 cm of its target 97% of
    the time. The skill does not retreat after placing, so it leaves the arm over the block.

    :param pixel: Raw (u,v) pixel coordinate indicating where to place the block
    :return: Updated image observation after the skill has finished
    """,
    #
    #
    "rotate": """Rotate the block at the given pixel in the current image.

    Select the target block's center pixel when calling `rotate`.

    Precondition: The block to be rotated cannot be in a stack and must be resting on the table.
        After calling `flip`, make sure to open the robot's gripper before calling `rotate`.

    Effect: `rotate` turns the target block a quarter turn in the requested direction,
    leaving the same face on top, and then releases the block.

    :param pixel: Raw (u,v) pixel coordinate of the block to rotate
    :param direction: Direction to rotate the block, as viewed from above ("cw" or "ccw")
    :return: Updated image observation after the skill has finished
    """,
    #
    #
    "flip": """Flip the block at the given pixel in the current image.

    Select the target block's center pixel when calling `flip`.

    Flip directions are image-relative. For example, the "front" face of a block is the
    one facing the camera, and the "left" face is on the left side of a block.

    When `flip` fails, it typically has little effect on the target block, so you can
    just call the skill again to retry when necessary.

    Precondition: The block to be flipped cannot be in a stack and must be resting on the table.

    Effect: The targeted block will be flipped such that the face that was on its top ends
        up as the block face specified by the `direction` argument. When a block is flipped
        left or right, its front face will rotate counterclockwise or clockwise, respectively.
    
    If you plan to call `rotate` after `flip`, open the gripper first.

    :param pixel: Raw (u,v) pixel coordinate of the block to flip
    :param direction: Specifies where the block's top face ends up ("front", "back", "left", "right")
    :return: Updated image observation after the skill has finished
    """,
    #
    #
    "return_home": """Move the arm back to a neutral resting configuration.

    Useful when the arm is blocking the camera's view. Never call `return_home` while holding a block.

    Effects: Leaves the state of the gripper (i.e., open, closed, or holding a block) unchanged.

    :return: Updated image observation after the skill has finished
    """,
    #
    #
    "open_gripper": """Open the robot's gripper.

    Effects: If the gripper is holding a block, it will be dropped.

    :return: Updated image observation after the skill has finished
    """,
    #
    #
    "close_gripper": """Close the robot's gripper.

    Effects: If a block is between the gripper fingertips, it will be grasped.

    :return: Updated image observation after the skill has finished
    """,
}
