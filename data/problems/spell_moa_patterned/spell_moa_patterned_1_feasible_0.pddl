(define (problem spell_moa_patterned_1_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_m block_o block_a block_q block_n block_v block_s - block_letter
    block_orange - block_orange
    block_yellow - block_yellow
    block_blue - block_blue
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_m workspace_table)
        (on block_o block_s)
        (on block_a block_m)
        (on_table block_q workspace_table)
        (on block_n block_q)
        (on_table block_v workspace_table)
        (on_table block_s workspace_table)
        (on_table block_orange workspace_table)
        (on_table block_yellow workspace_table)
        (on block_blue block_yellow)
        (clear block_o)
        (clear block_a)
        (clear block_n)
        (clear block_v)
        (clear block_orange)
        (clear block_blue)
        (ori_of block_m o23)
        (ori_of block_o o5)
        (ori_of block_a o3)
        (ori_of block_q o10)
        (ori_of block_n o17)
        (ori_of block_v o21)
        (ori_of block_s o2)
        (ori_of block_orange o20)
        (ori_of block_yellow o5)
        (ori_of block_blue o13)
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
