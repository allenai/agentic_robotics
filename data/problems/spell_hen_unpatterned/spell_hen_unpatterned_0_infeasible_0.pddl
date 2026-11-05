(define (problem spell_hen_unpatterned_0_infeasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00029 block_00012 block_00010 block_00015 block_00020 block_00032 - block_letter
    block_yellow - block_yellow
    block_red1 - block_red
    block_orange - block_orange
    block_blue - block_blue
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_00029 block_yellow)
        (on_table block_00012 workspace_table)
        (on_table block_00010 workspace_table)
        (on_table block_00015 workspace_table)
        (on_table block_00020 workspace_table)
        (on block_00032 block_00015)
        (on_table block_yellow workspace_table)
        (on_table block_red1 workspace_table)
        (on_table block_orange workspace_table)
        (on block_blue block_00020)
        (clear block_00029)
        (clear block_00012)
        (clear block_00010)
        (clear block_00032)
        (clear block_red1)
        (clear block_orange)
        (clear block_blue)
        (ori_of block_00029 o24)
        (ori_of block_00012 o15)
        (ori_of block_00010 o14)
        (ori_of block_00015 o22)
        (ori_of block_00020 o24)
        (ori_of block_00032 o24)
        (ori_of block_yellow o8)
        (ori_of block_red1 o17)
        (ori_of block_orange o16)
        (ori_of block_blue o14)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00015 workspace_table)
        (on block_00012 block_00015)
        (on block_00029 block_00012)
        (clear block_00029)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
