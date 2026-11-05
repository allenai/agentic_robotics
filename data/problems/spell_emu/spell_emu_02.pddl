(define (problem spell_emu_02)
    (:domain blocks-tamp)
    (:objects
    block_e block_m block_u block_n block_o block_q block_d - block_letter
    block_yellow block_red1 block_green block_blue - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_e workspace_table)
        (on_table block_m workspace_table)
        (on_table block_u workspace_table)
        (on block_n block_u)
        (clear block_n)
        (on_table block_o workspace_table)
        (clear block_o)
        (on_table block_q workspace_table)
        (clear block_q)
        (on_table block_d workspace_table)
        (clear block_d)
        (on block_yellow block_e)
        (clear block_yellow)
        (on_table block_red1 workspace_table)
        (clear block_red1)
        (on block_green block_m)
        (clear block_green)
        (on_table block_blue workspace_table)
        (clear block_blue)
        (ori_of block_e o20)
        (ori_of block_m o21)
        (ori_of block_u o18)
        (ori_of block_n o2)
        (ori_of block_o o1)
        (ori_of block_q o20)
        (ori_of block_d o23)
        (ori_of block_yellow o13)
        (ori_of block_red1 o15)
        (ori_of block_green o9)
        (ori_of block_blue o4)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (ori_of block_e o2)
        (ori_of block_m o2)
        (ori_of block_u o2)
    ))

    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [0, 0.04]
    ;friction_range = [0.1, 0.4]
)