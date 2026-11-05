(define (problem spell_emu_unpatterned_1_infeasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00006 block_00012 block_00030 block_00032 block_00035 block_00034 block_00014 - block_letter
    block_blue - block_blue
    block_red1 - block_red
    block_green - block_green
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_00006 workspace_table)
        (on block_00012 block_00032)
        (on block_00030 block_green)
        (on_table block_00032 workspace_table)
        (on block_00035 block_00006)
        (on_table block_00034 workspace_table)
        (on_table block_00014 workspace_table)
        (on block_blue block_red1)
        (on_table block_red1 workspace_table)
        (on_table block_green workspace_table)
        (clear block_00012)
        (clear block_00030)
        (clear block_00035)
        (clear block_00034)
        (clear block_00014)
        (clear block_blue)
        (ori_of block_00006 o8)
        (ori_of block_00012 o18)
        (ori_of block_00030 o14)
        (ori_of block_00032 o23)
        (ori_of block_00035 o5)
        (ori_of block_00034 o1)
        (ori_of block_00014 o15)
        (ori_of block_blue o7)
        (ori_of block_red1 o21)
        (ori_of block_green o3)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00034 workspace_table)
        (on block_00006 block_00034)
        (on block_00012 block_00006)
        (clear block_00012)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
