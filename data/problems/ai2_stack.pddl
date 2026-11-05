(define (problem ai2_stack)
    (:domain blocks-tamp)
    ; A, I and 2, plus three distractors: one lettered and two coloured. The lettered distractor is
    ; never only what it looks like -- block_j is the J_W block, and every glyph block carries the
    ; glyph thirteen along the alphabet from its own on the opposite face -- so a distractor's
    ; hidden faces are part of what the scene offers.
    ; None of the three targets starts readable: face five carries a block's named glyph and is
    ; forward only at o2, o6, o10 and o14.
    (:objects
    block_a - block_a_n
    block_i - block_i_v
    block_2 - block_2_6
    block_j - block_j_w
    block_green block_blue - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_a workspace_table)
        (on_table block_i workspace_table)
        (on_table block_2 workspace_table)
        (on_table block_j workspace_table)
        (on_table block_green workspace_table)
        (on_table block_blue workspace_table)

        (clear block_a)
        (clear block_i)
        (clear block_2)
        (clear block_j)
        (clear block_green)
        (clear block_blue)

        (handempty robot)

        (ori_of block_a o8)
        (ori_of block_i o1)
        (ori_of block_2 o4)
        (ori_of block_j o1)
        (ori_of block_green o1)
        (ori_of block_blue o1)

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
