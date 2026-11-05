(define (problem spell_moa_patterned_2_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_m block_o block_a block_e block_c block_q block_g block_n - block_letter
    block_purple - block_purple
    block_green - block_green
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_m workspace_table)
        (on_table block_o workspace_table)
        (on block_a block_purple)
        (on block_e block_m)
        (on_table block_c workspace_table)
        (on_table block_q workspace_table)
        (on block_g block_green)
        (on block_n block_c)
        (on_table block_purple workspace_table)
        (on_table block_green workspace_table)
        (clear block_o)
        (clear block_a)
        (clear block_e)
        (clear block_q)
        (clear block_g)
        (clear block_n)
        (ori_of block_m o9)
        (ori_of block_o o15)
        (ori_of block_a o7)
        (ori_of block_e o12)
        (ori_of block_c o18)
        (ori_of block_q o14)
        (ori_of block_g o4)
        (ori_of block_n o7)
        (ori_of block_purple o11)
        (ori_of block_green o24)
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
