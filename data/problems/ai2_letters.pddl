(define (problem ai2_letters)
    (:domain blocks-tamp)
    ; Ai2 stack, letters hidden. 3 targets plus 3 lettered distractors and 6 colour distractors: 12 blocks.
    ; The lettered distractors are R, T and J. Their 13-apart partners are E, G and W,
    ; so no distractor is physically an A, I or 2 block -- a scene with `block_n` would
    ; hold a second A, since `block_n` and `block_a` are the same physical block.
    ; None of the lettered blocks starts readable: face five carries a block's named glyph and
    ; is forward only at o2, o6, o10 and o14, so every letter has to be found by turning.
    ; The goal below is only what Stage-1 codegen consumes; the agent gets its prompt.
    (:objects
    block_a - block_a_n
    block_i - block_i_v
    block_2 - block_2_6
    block_r - block_r_e
    block_t - block_t_g
    block_j - block_j_w
    block_red1 block_green block_blue block_yellow block_orange block_purple - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_a workspace_table)
        (on_table block_i workspace_table)
        (on_table block_2 workspace_table)
        (on_table block_r workspace_table)
        (on_table block_t workspace_table)
        (on_table block_j workspace_table)
        (on_table block_red1 workspace_table)
        (on_table block_green workspace_table)
        (on_table block_blue workspace_table)
        (on_table block_yellow workspace_table)
        (on_table block_orange workspace_table)
        (on_table block_purple workspace_table)

        (clear block_a)
        (clear block_i)
        (clear block_2)
        (clear block_r)
        (clear block_t)
        (clear block_j)
        (clear block_red1)
        (clear block_green)
        (clear block_blue)
        (clear block_yellow)
        (clear block_orange)
        (clear block_purple)

        (handempty robot)

        (ori_of block_a o8)
        (ori_of block_i o1)
        (ori_of block_2 o4)
        (ori_of block_r o8)
        (ori_of block_t o4)
        (ori_of block_j o1)
        (ori_of block_red1 o1)
        (ori_of block_green o1)
        (ori_of block_blue o1)
        (ori_of block_yellow o1)
        (ori_of block_orange o1)
        (ori_of block_purple o1)

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
