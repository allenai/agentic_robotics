(define (problem spell_emu_patterned_2_infeasible_1)
    (:domain blocks-tamp)
    (:objects
    block_e block_m block_o block_q block_c block_k block_g block_d - block_letter
    block_purple - block_purple
    block_green - block_green
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_e block_purple)
        (on_table block_m workspace_table)
        (on_table block_o workspace_table)
        (on_table block_q workspace_table)
        (on_table block_c workspace_table)
        (on block_k block_c)
        (on block_g block_q)
        (on_table block_d workspace_table)
        (on_table block_purple workspace_table)
        (on block_green block_o)
        (clear block_e)
        (clear block_m)
        (clear block_k)
        (clear block_g)
        (clear block_d)
        (clear block_green)
        (ori_of block_e o13)
        (ori_of block_m o15)
        (ori_of block_o o9)
        (ori_of block_q o8)
        (ori_of block_c o16)
        (ori_of block_k o15)
        (ori_of block_g o23)
        (ori_of block_d o11)
        (ori_of block_purple o21)
        (ori_of block_green o15)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_o workspace_table)
        (on block_m block_o)
        (on block_e block_m)
        (clear block_e)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
