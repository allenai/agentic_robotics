"""Define oracle docstrings for a baseline agent."""

ORACLE_DOCSTRINGS: dict[str, str] = {
    "pick": """Pick up the block at the given pixel in the current image.

    Preconditions:
        - The robot's gripper must already be open.
        - A block cannot be picked if it is within ~3 cm of a stack of blocks. First, use
            the `flip` skill to move the block away from the stack, or take the stack apart
            from the top. Single blocks resting on the table nearby do not prevent a pick.
        - Only the top block in a stack can be successfully picked. Attempting to pick
            any other block in the stack will result in the top block being picked.

    Effects:
        - The robot's gripper will close on the block and lift it from the table by 20 cm.
        - If `pick` is called when its preconditions aren't met, the skill will
            move the end-effector above the target block and close on nothing. Thus,
            the next `pick` will also fail until the gripper is opened.

    :param pixel: Raw (u,v) pixel coordinate of the block to pick
    :return: Updated image observation after the skill has finished
    """,
    #
    #
    "stack": """Stack the held block onto the block at the given pixel in the current image.

    There is a 10% chance that this skill will drop the block 6 cm above its target location.

    Sometimes, if `stack` is unable to identify a motion plan, it will do nothing
    and exit, leaving the held block in the gripper. In such cases, try executing
    a different skill to free up motion planning.

    Preconditions: The robot must already be holding a block.

    Effects:
        - The robot's gripper is opened and then retracted from the stacked block.
        - If the robot was not holding a block, this skill will have no effect.

    :param pixel: Raw (u,v) pixel coordinate of the block to be stacked on
    :return: Updated image observation after the skill has finished
    """,
    #
    #
    "place_on_table": """Place the held block onto the table, at the given pixel in the current image.

    There is a 10% chance that this skill will drop the block 6 cm above its target location.

    Sometimes, if `place_on_table` is unable to identify a motion plan, it will do
    nothing and exit, leaving the held block in the gripper. In such cases, try
    executing a different skill to free up motion planning.

    Preconditions: The robot must already be holding a block.

    Effects:
        - The robot's gripper is opened and then retracted from the placed block.
        - If the robot was not holding a block, this skill will have no effect.

    :param pixel: Raw (u,v) pixel coordinate indicating where to place the block
    :return: Updated image observation after the skill has finished
    """,
    #
    #
    "rotate": """Rotate the block at the given pixel in the current image.

    Avoid rotating a block when it's within ~5 cm of a tower of blocks; they may get knocked over.
    You can use the `flip` skill to move a block away from a nearby tower.

    There is a 25% chance that the block will be rotated the wrong way. Double-check after.

    Preconditions:
        - The robot's gripper must be empty.
        - The block to be rotated must be resting on the table, not stacked with other blocks.
        - The block to be rotated must be in a special central area of the table, which excludes
            the far left side, far right side, and the strip of the table nearest the robot.

    Effects:
        - The robot's gripper ends up open.
        - The block will be rotated to the next axis-aligned (w.r.t. the camera) orientation.
        - If any precondition is not met, `rotate` will bring the end-effector
            above the target block but will not rotate the block.

    :param pixel: Raw (u,v) pixel coordinate of the block to rotate
    :param direction: Direction to rotate the block, as viewed from above ("cw" or "ccw")
    :return: Updated image observation after the skill has finished
    """,
    #
    #
    "flip": """Flip the block at the given pixel in the current image.

    Flip works when the target block is beside an unstacked block, but may disturb nearby stacks.
    You can use this skill to separate a single block from its neighbor, because a
    flipped block is moved about 1-3 cm in the flip direction.

    Flipping a block toward a nearby stack usually knocks the stack over; flip away from it.

    There is a 10% chance that this skill will have no effect, and a 20% chance
    that the block will be flipped in another direction.

    Preconditions:
        - The robot's gripper must be empty.
        - The block to be flipped must be resting on the table, not stacked with other blocks.
        - The block to be flipped must be in a special central area of the table, which excludes
            the far left side, far right side, and the strip of the table nearest the robot.

    Effects:
        - The robot's gripper ends up closed.
        - The block will be flipped so that its top face ends up facing the specified direction.
        - A left or right flip turns the camera-facing front block face a quarter turn,
            so an upright letter ends up lying on its side, still on the front face.
        - If any precondition is not met, `flip` will bring the end-effector
            above the target block but will not flip the block.

    :param pixel: Raw (u,v) pixel coordinate of the block to flip
    :param direction: Specifies where the block's top face ends up ("front", "back", "left", "right")
    :return: Updated image observation after the skill has finished
    """,
    #
    #
    "return_home": """Move the arm back to a neutral resting configuration.

    Useful when the arm is blocking the camera's view.

    Effects: The gripper and anything it holds are left as they were.

    :return: Updated image observation after the skill has finished
    """,
    #
    #
    "open_gripper": """Open the robot's gripper.

    Effects: Any held blocks will be dropped.

    :return: Updated image observation after the skill has finished
    """,
    #
    #
    "close_gripper": """Close the robot's gripper.

    Effects: If a block is between the gripper fingertips, it will be grasped.

    :return: Updated image observation after the skill has finished
    """,
}
