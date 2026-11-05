(define (problem spell_auk_patterned_1_infeasible_1)
    (:domain blocks-tamp)
    (:objects
    block_a block_d block_k block_g block_v block_t block_p - block_letter
    block_red1 - block_red
    block_yellow - block_yellow
    block_purple - block_purple
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_a workspace_table)
        (on_table block_d workspace_table)
        (on_table block_k workspace_table)
        (on_table block_g workspace_table)
        (on_table block_v workspace_table)
        (on_table block_t workspace_table)
        (on block_p block_red1)
        (on_table block_red1 workspace_table)
        (on_table block_yellow workspace_table)
        (on block_purple block_k)
        (clear block_a)
        (clear block_d)
        (clear block_g)
        (clear block_v)
        (clear block_t)
        (clear block_p)
        (clear block_yellow)
        (clear block_purple)
        (ori_of block_a o5)
        (ori_of block_d o17)
        (ori_of block_k o13)
        (ori_of block_g o11)
        (ori_of block_v o8)
        (ori_of block_t o13)
        (ori_of block_p o2)
        (ori_of block_red1 o9)
        (ori_of block_yellow o9)
        (ori_of block_purple o3)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_k workspace_table)
        (on block_d block_k)
        (on block_a block_d)
        (clear block_a)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
