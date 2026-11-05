(define (problem spell_moa_unpatterned_0_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00001 block_00011 block_00021 block_00027 block_00030 block_00033 - block_letter
    block_red1 - block_red
    block_blue - block_blue
    block_yellow - block_yellow
    block_purple - block_purple
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_00001 workspace_table)
        (on block_00011 block_00021)
        (on_table block_00021 workspace_table)
        (on_table block_00027 workspace_table)
        (on block_00030 block_00001)
        (on block_00033 block_00027)
        (on_table block_red1 workspace_table)
        (on_table block_blue workspace_table)
        (on_table block_yellow workspace_table)
        (on block_purple block_blue)
        (clear block_00011)
        (clear block_00030)
        (clear block_00033)
        (clear block_red1)
        (clear block_yellow)
        (clear block_purple)
        (ori_of block_00001 o9)
        (ori_of block_00011 o7)
        (ori_of block_00021 o4)
        (ori_of block_00027 o7)
        (ori_of block_00030 o12)
        (ori_of block_00033 o1)
        (ori_of block_red1 o12)
        (ori_of block_blue o5)
        (ori_of block_yellow o20)
        (ori_of block_purple o14)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00027 workspace_table)
        (on block_00001 block_00027)
        (on block_00011 block_00001)
        (clear block_00011)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
