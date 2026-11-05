(define (problem spell_null_00)
    (:domain blocks-tamp)
    (:objects
    block_n block_u block_l block_l_2 block_y block_d block_o block_m block_c - block_letter
    block_blue block_orange block_yellow - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_n workspace_table)
        (on_table block_u workspace_table)
        (clear block_u)
        (on_table block_l workspace_table)
        (on_table block_l_2 workspace_table)
        (on block_y block_l_2)
        (clear block_y)
        (on_table block_d workspace_table)
        (clear block_d)
        (on_table block_o workspace_table)
        (clear block_o)
        (on_table block_m workspace_table)
        (clear block_m)
        (on_table block_c workspace_table)
        (clear block_c)
        (on block_blue block_l)
        (clear block_blue)
        (on block_orange block_n)
        (clear block_orange)
        (on_table block_yellow workspace_table)
        (clear block_yellow)
        (ori_of block_n o20)
        (ori_of block_u o11)
        (ori_of block_l o24)
        (ori_of block_l_2 o20)
        (ori_of block_y o22)
        (ori_of block_d o4)
        (ori_of block_o o13)
        (ori_of block_m o12)
        (ori_of block_c o3)
        (ori_of block_blue o15)
        (ori_of block_orange o6)
        (ori_of block_yellow o18)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (ori_of block_n o2)
        (ori_of block_u o2)
        (ori_of block_l o2)
    ))

    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [0, 0.04]
    ;friction_range = [0.1, 0.4]
)