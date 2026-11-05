(define (problem spell_hen_patterned_0_infeasible_0)
    (:domain blocks-tamp)
    (:objects
    block_m block_e block_n block_y block_l block_f - block_letter
    block_yellow - block_yellow
    block_red1 - block_red
    block_orange - block_orange
    block_blue - block_blue
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_m block_yellow)
        (on_table block_e workspace_table)
        (on_table block_n workspace_table)
        (on_table block_y workspace_table)
        (on_table block_l workspace_table)
        (on block_f block_y)
        (on_table block_yellow workspace_table)
        (on_table block_red1 workspace_table)
        (on_table block_orange workspace_table)
        (on block_blue block_l)
        (clear block_m)
        (clear block_e)
        (clear block_n)
        (clear block_f)
        (clear block_red1)
        (clear block_orange)
        (clear block_blue)
        (ori_of block_m o2)
        (ori_of block_e o1)
        (ori_of block_n o9)
        (ori_of block_y o16)
        (ori_of block_l o6)
        (ori_of block_f o4)
        (ori_of block_yellow o11)
        (ori_of block_red1 o13)
        (ori_of block_orange o2)
        (ori_of block_blue o10)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_n workspace_table)
        (on block_e block_n)
        (on block_m block_e)
        (clear block_m)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
