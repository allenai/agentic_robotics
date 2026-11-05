(define (problem spell_moa_unpatterned_1_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00001 block_00011 block_00021 block_00027 block_00030 block_00033 block_00002 - block_letter
    block_orange - block_orange
    block_yellow - block_yellow
    block_blue - block_blue
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_00001 workspace_table)
        (on block_00011 block_00002)
        (on block_00021 block_00001)
        (on_table block_00027 workspace_table)
        (on block_00030 block_00027)
        (on_table block_00033 workspace_table)
        (on_table block_00002 workspace_table)
        (on_table block_orange workspace_table)
        (on_table block_yellow workspace_table)
        (on block_blue block_yellow)
        (clear block_00011)
        (clear block_00021)
        (clear block_00030)
        (clear block_00033)
        (clear block_orange)
        (clear block_blue)
        (ori_of block_00001 o22)
        (ori_of block_00011 o1)
        (ori_of block_00021 o16)
        (ori_of block_00027 o16)
        (ori_of block_00030 o15)
        (ori_of block_00033 o8)
        (ori_of block_00002 o2)
        (ori_of block_orange o4)
        (ori_of block_yellow o23)
        (ori_of block_blue o9)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00021 workspace_table)
        (on block_00030 block_00021)
        (on block_00011 block_00030)
        (clear block_00011)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
