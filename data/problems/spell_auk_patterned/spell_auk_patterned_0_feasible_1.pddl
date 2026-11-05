(define (problem spell_auk_patterned_0_feasible_1)
    (:domain blocks-tamp)
    (:objects
    block_a block_u block_k block_d block_r block_s - block_letter
    block_red1 - block_red
    block_yellow - block_yellow
    block_orange - block_orange
    block_blue - block_blue
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_a workspace_table)
        (on_table block_u workspace_table)
        (on block_k block_u)
        (on_table block_d workspace_table)
        (on block_r block_blue)
        (on_table block_s workspace_table)
        (on_table block_red1 workspace_table)
        (on_table block_yellow workspace_table)
        (on_table block_orange workspace_table)
        (on_table block_blue workspace_table)
        (clear block_a)
        (clear block_k)
        (clear block_d)
        (clear block_r)
        (clear block_s)
        (clear block_red1)
        (clear block_yellow)
        (clear block_orange)
        (ori_of block_a o15)
        (ori_of block_u o1)
        (ori_of block_k o5)
        (ori_of block_d o2)
        (ori_of block_r o6)
        (ori_of block_s o21)
        (ori_of block_red1 o24)
        (ori_of block_yellow o17)
        (ori_of block_orange o12)
        (ori_of block_blue o24)
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
