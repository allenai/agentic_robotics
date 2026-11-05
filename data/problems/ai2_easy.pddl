(define (problem ai2_easy)
    (:domain blocks-tamp)
    ; The Ai2 stack with nothing hidden: only A, I and 2 on the table, no distractors, and every
    ; block already at o2 -- the orientation the goal asks for, and the one where face five's
    ; named glyph faces the camera. Reading and reorienting are both off the table, so what is
    ; left is finding each block and stacking A on I on 2 without knocking an orientation off.
    (:objects
    block_a - block_a_n
    block_i - block_i_v
    block_2 - block_2_6
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_a workspace_table)
        (on_table block_i workspace_table)
        (on_table block_2 workspace_table)

        (clear block_a)
        (clear block_i)
        (clear block_2)

        (handempty robot)

        (ori_of block_a o2)
        (ori_of block_i o2)
        (ori_of block_2 o2)

        (gripper_at_init)
    )
    (:goal (and
        (on block_a block_i)
        (on block_i block_2)
        (ori_of block_a o2)
        (ori_of block_i o2)
        (ori_of block_2 o2)
    ))
)
