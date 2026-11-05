(define (problem spell_emu_patterned_1_infeasible_0)
    (:domain blocks-tamp)
    (:objects
    block_e block_q block_u block_t block_i block_a block_w - block_letter
    block_blue - block_blue
    block_red1 - block_red
    block_green - block_green
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_e workspace_table)
        (on block_q block_t)
        (on block_u block_green)
        (on_table block_t workspace_table)
        (on block_i block_e)
        (on_table block_a workspace_table)
        (on_table block_w workspace_table)
        (on block_blue block_red1)
        (on_table block_red1 workspace_table)
        (on_table block_green workspace_table)
        (clear block_q)
        (clear block_u)
        (clear block_i)
        (clear block_a)
        (clear block_w)
        (clear block_blue)
        (ori_of block_e o21)
        (ori_of block_q o9)
        (ori_of block_u o9)
        (ori_of block_t o24)
        (ori_of block_i o4)
        (ori_of block_a o6)
        (ori_of block_w o6)
        (ori_of block_blue o6)
        (ori_of block_red1 o13)
        (ori_of block_green o6)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_u workspace_table)
        (on block_q block_u)
        (on block_e block_q)
        (clear block_e)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
