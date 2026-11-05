(define (problem spell_auk_patterned_2_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_a block_u block_k block_y block_w block_r block_o block_z - block_letter
    block_orange - block_orange
    block_red1 - block_red
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_a workspace_table)
        (on block_u block_y)
        (on_table block_k workspace_table)
        (on_table block_y workspace_table)
        (on_table block_w workspace_table)
        (on_table block_r workspace_table)
        (on_table block_o workspace_table)
        (on_table block_z workspace_table)
        (on block_orange block_red1)
        (on_table block_red1 workspace_table)
        (clear block_a)
        (clear block_u)
        (clear block_k)
        (clear block_w)
        (clear block_r)
        (clear block_o)
        (clear block_z)
        (clear block_orange)
        (ori_of block_a o5)
        (ori_of block_u o1)
        (ori_of block_k o11)
        (ori_of block_y o23)
        (ori_of block_w o10)
        (ori_of block_r o1)
        (ori_of block_o o7)
        (ori_of block_z o11)
        (ori_of block_orange o10)
        (ori_of block_red1 o11)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_k workspace_table)
        (on block_u block_k)
        (on block_a block_u)
        (clear block_a)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
