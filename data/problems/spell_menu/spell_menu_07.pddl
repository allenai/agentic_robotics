(define (problem spell_menu_07)
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
        (clear block_e)
        (on_table block_n workspace_table)
        (clear block_n)
        (on_table block_u workspace_table)
        (clear block_u)
        (on_table block_b workspace_table)
        (clear block_b)
        (on_table block_s workspace_table)
        (clear block_s)
        (on_table block_j workspace_table)
        (clear block_j)
        (on_table block_g workspace_table)
        (clear block_g)
        (on_table block_w workspace_table)
        (clear block_w)
        (on_table block_yellow workspace_table)
        (clear block_yellow)
        (ori_of block_m o22)
        (ori_of block_e o3)
        (ori_of block_n o14)
        (ori_of block_u o20)
        (ori_of block_b o23)
        (ori_of block_s o12)
        (ori_of block_j o6)
        (ori_of block_g o18)
        (ori_of block_w o23)
        (ori_of block_yellow o6)
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