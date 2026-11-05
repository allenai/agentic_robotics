(define (problem spell_emu_patterned_0_feasible_1)
    (:domain blocks-tamp)
    (:objects
    block_e block_m block_u block_c block_i block_n - block_letter
    block_green - block_green
    block_red1 - block_red
    block_orange - block_orange
    block_yellow - block_yellow
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_e workspace_table)
        (on block_m block_c)
        (on block_u block_orange)
        (on_table block_c workspace_table)
        (on_table block_i workspace_table)
        (on block_n block_red1)
        (on block_green block_e)
        (on_table block_red1 workspace_table)
        (on_table block_orange workspace_table)
        (on_table block_yellow workspace_table)
        (clear block_m)
        (clear block_u)
        (clear block_i)
        (clear block_n)
        (clear block_green)
        (clear block_yellow)
        (ori_of block_e o17)
        (ori_of block_m o11)
        (ori_of block_u o13)
        (ori_of block_c o3)
        (ori_of block_i o16)
        (ori_of block_n o21)
        (ori_of block_green o16)
        (ori_of block_red1 o18)
        (ori_of block_orange o19)
        (ori_of block_yellow o18)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_u workspace_table)
        (on block_m block_u)
        (on block_e block_m)
        (clear block_e)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
