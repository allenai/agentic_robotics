(define (problem spell_hen_patterned_1_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_h block_e block_n block_c block_d block_m block_x - block_letter
    block_yellow - block_yellow
    block_blue - block_blue
    block_red1 - block_red
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_h block_blue)
        (on_table block_e workspace_table)
        (on_table block_n workspace_table)
        (on_table block_c workspace_table)
        (on_table block_d workspace_table)
        (on_table block_m workspace_table)
        (on_table block_x workspace_table)
        (on block_yellow block_e)
        (on_table block_blue workspace_table)
        (on block_red1 block_d)
        (clear block_h)
        (clear block_n)
        (clear block_c)
        (clear block_m)
        (clear block_x)
        (clear block_yellow)
        (clear block_red1)
        (ori_of block_h o9)
        (ori_of block_e o20)
        (ori_of block_n o15)
        (ori_of block_c o18)
        (ori_of block_d o6)
        (ori_of block_m o8)
        (ori_of block_x o10)
        (ori_of block_yellow o22)
        (ori_of block_blue o1)
        (ori_of block_red1 o14)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_n workspace_table)
        (on block_e block_n)
        (on block_h block_e)
        (clear block_h)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
