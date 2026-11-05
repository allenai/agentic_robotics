(define (problem spell_moa_patterned_4_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_m block_o block_a block_p block_u block_g block_r block_w block_y block_z - block_letter
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_m workspace_table)
        (on block_o block_m)
        (on_table block_a workspace_table)
        (on_table block_p workspace_table)
        (on block_u block_a)
        (on_table block_g workspace_table)
        (on block_r block_w)
        (on_table block_w workspace_table)
        (on block_y block_g)
        (on_table block_z workspace_table)
        (clear block_o)
        (clear block_p)
        (clear block_u)
        (clear block_r)
        (clear block_y)
        (clear block_z)
        (ori_of block_m o13)
        (ori_of block_o o15)
        (ori_of block_a o5)
        (ori_of block_p o17)
        (ori_of block_u o3)
        (ori_of block_g o17)
        (ori_of block_r o5)
        (ori_of block_w o7)
        (ori_of block_y o5)
        (ori_of block_z o22)
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
