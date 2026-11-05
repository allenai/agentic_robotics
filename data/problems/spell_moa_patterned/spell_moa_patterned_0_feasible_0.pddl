(define (problem spell_moa_patterned_0_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_m block_o block_a block_j block_b block_z - block_letter
    block_red1 - block_red
    block_blue - block_blue
    block_yellow - block_yellow
    block_purple - block_purple
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_m workspace_table)
        (on block_o block_a)
        (on_table block_a workspace_table)
        (on_table block_j workspace_table)
        (on block_b block_m)
        (on block_z block_j)
        (on_table block_red1 workspace_table)
        (on_table block_blue workspace_table)
        (on_table block_yellow workspace_table)
        (on block_purple block_blue)
        (clear block_o)
        (clear block_b)
        (clear block_z)
        (clear block_red1)
        (clear block_yellow)
        (clear block_purple)
        (ori_of block_m o24)
        (ori_of block_o o7)
        (ori_of block_a o3)
        (ori_of block_j o16)
        (ori_of block_b o4)
        (ori_of block_z o1)
        (ori_of block_red1 o8)
        (ori_of block_blue o19)
        (ori_of block_yellow o21)
        (ori_of block_purple o12)
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
