(define (problem spell_hen_unpatterned_2_infeasible_1)
    (:domain blocks-tamp)
    (:objects
    block_00002 block_00012 block_00013 block_00022 block_00026 block_00032 block_00028 block_00007 - block_letter
    block_blue - block_blue
    block_purple - block_purple
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_00002 workspace_table)
        (on_table block_00012 workspace_table)
        (on_table block_00013 workspace_table)
        (on_table block_00022 workspace_table)
        (on block_00026 block_purple)
        (on_table block_00032 workspace_table)
        (on block_00028 block_00022)
        (on block_00007 block_blue)
        (on_table block_blue workspace_table)
        (on_table block_purple workspace_table)
        (clear block_00002)
        (clear block_00012)
        (clear block_00013)
        (clear block_00026)
        (clear block_00032)
        (clear block_00028)
        (clear block_00007)
        (ori_of block_00002 o20)
        (ori_of block_00012 o20)
        (ori_of block_00013 o12)
        (ori_of block_00022 o16)
        (ori_of block_00026 o21)
        (ori_of block_00032 o7)
        (ori_of block_00028 o12)
        (ori_of block_00007 o16)
        (ori_of block_blue o10)
        (ori_of block_purple o4)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00022 workspace_table)
        (on block_00032 block_00022)
        (on block_00013 block_00032)
        (clear block_00013)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
