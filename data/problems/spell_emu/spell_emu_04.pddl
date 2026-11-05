(define (problem spell_emu_04)
    (:domain blocks-tamp)
    (:objects
    block_e block_m block_u block_n block_o block_q block_d - block_letter
    block_yellow block_red1 block_green block_blue - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_e workspace_table)
        (on_table block_m workspace_table)
        (clear block_m)
        (on_table block_u workspace_table)
        (on_table block_n workspace_table)
        (clear block_n)
        (on_table block_o workspace_table)
        (clear block_o)
        (on block_q block_u)
        (clear block_q)
        (on_table block_d workspace_table)
        (clear block_d)
        (on_table block_yellow workspace_table)
        (clear block_yellow)
        (on block_red1 block_e)
        (clear block_red1)
        (on_table block_green workspace_table)
        (clear block_green)
        (on_table block_blue workspace_table)
        (clear block_blue)
        (ori_of block_e o18)
        (ori_of block_m o5)
        (ori_of block_u o17)
        (ori_of block_n o9)
        (ori_of block_o o11)
        (ori_of block_q o20)
        (ori_of block_d o19)
        (ori_of block_yellow o5)
        (ori_of block_red1 o16)
        (ori_of block_green o9)
        (ori_of block_blue o2)
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