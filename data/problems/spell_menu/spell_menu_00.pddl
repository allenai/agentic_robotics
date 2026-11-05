(define (problem spell_menu_00)
    (:domain blocks-tamp)
    (:objects
    block_m block_e block_n block_u block_b block_s block_j block_g block_w - block_letter
    block_yellow - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_m workspace_table)
        (clear block_m)
        (on_table block_e workspace_table)
        (on_table block_n workspace_table)
        (clear block_n)
        (on_table block_u workspace_table)
        (on block_b block_u)
        (clear block_b)
        (on_table block_s workspace_table)
        (clear block_s)
        (on_table block_j workspace_table)
        (clear block_j)
        (on block_g block_e)
        (clear block_g)
        (on_table block_w workspace_table)
        (clear block_w)
        (on_table block_yellow workspace_table)
        (clear block_yellow)
        (ori_of block_m o5)
        (ori_of block_e o22)
        (ori_of block_n o11)
        (ori_of block_u o23)
        (ori_of block_b o2)
        (ori_of block_s o16)
        (ori_of block_j o9)
        (ori_of block_g o23)
        (ori_of block_w o6)
        (ori_of block_yellow o19)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (ori_of block_m o2)
        (ori_of block_e o2)
        (ori_of block_n o2)
    ))

    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [0, 0.04]
    ;friction_range = [0.1, 0.4]
)