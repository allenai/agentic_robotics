(define (problem spell_moa_patterned_3_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_m block_o block_a block_f block_z block_v block_p block_i block_q - block_letter
    block_blue - block_blue
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_m workspace_table)
        (on block_o block_p)
        (on_table block_a workspace_table)
        (on block_f block_z)
        (on_table block_z workspace_table)
        (on_table block_v workspace_table)
        (on_table block_p workspace_table)
        (on block_i block_a)
        (on_table block_q workspace_table)
        (on block_blue block_v)
        (clear block_m)
        (clear block_o)
        (clear block_f)
        (clear block_i)
        (clear block_q)
        (clear block_blue)
        (ori_of block_m o15)
        (ori_of block_o o5)
        (ori_of block_a o23)
        (ori_of block_f o13)
        (ori_of block_z o8)
        (ori_of block_v o15)
        (ori_of block_p o23)
        (ori_of block_i o9)
        (ori_of block_q o10)
        (ori_of block_blue o22)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_a workspace_table)
        (on block_o block_a)
        (on block_m block_o)
        (clear block_m)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
