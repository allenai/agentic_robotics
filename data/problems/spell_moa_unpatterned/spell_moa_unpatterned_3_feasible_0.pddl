(define (problem spell_moa_unpatterned_3_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00001 block_00011 block_00021 block_00027 block_00030 block_00033 block_00004 block_00035 block_00002 - block_letter
    block_blue - block_blue
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_00001 workspace_table)
        (on block_00011 block_00004)
        (on_table block_00021 workspace_table)
        (on block_00027 block_00030)
        (on_table block_00030 workspace_table)
        (on_table block_00033 workspace_table)
        (on_table block_00004 workspace_table)
        (on block_00035 block_00021)
        (on_table block_00002 workspace_table)
        (on block_blue block_00033)
        (clear block_00001)
        (clear block_00011)
        (clear block_00027)
        (clear block_00035)
        (clear block_00002)
        (clear block_blue)
        (ori_of block_00001 o11)
        (ori_of block_00011 o24)
        (ori_of block_00021 o23)
        (ori_of block_00027 o5)
        (ori_of block_00030 o14)
        (ori_of block_00033 o22)
        (ori_of block_00004 o1)
        (ori_of block_00035 o23)
        (ori_of block_00002 o11)
        (ori_of block_blue o21)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00021 workspace_table)
        (on block_00001 block_00021)
        (on block_00011 block_00001)
        (clear block_00011)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
