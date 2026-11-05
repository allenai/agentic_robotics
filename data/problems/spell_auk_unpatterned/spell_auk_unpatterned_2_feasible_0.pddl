(define (problem spell_auk_unpatterned_2_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00020 block_00021 block_00027 block_00030 block_00032 block_00034 block_00004 block_00002 - block_letter
    block_orange - block_orange
    block_red1 - block_red
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_00020 workspace_table)
        (on block_00021 block_00030)
        (on_table block_00027 workspace_table)
        (on_table block_00030 workspace_table)
        (on_table block_00032 workspace_table)
        (on_table block_00034 workspace_table)
        (on_table block_00004 workspace_table)
        (on_table block_00002 workspace_table)
        (on block_orange block_red1)
        (on_table block_red1 workspace_table)
        (clear block_00020)
        (clear block_00021)
        (clear block_00027)
        (clear block_00032)
        (clear block_00034)
        (clear block_00004)
        (clear block_00002)
        (clear block_orange)
        (ori_of block_00020 o10)
        (ori_of block_00021 o9)
        (ori_of block_00027 o11)
        (ori_of block_00030 o16)
        (ori_of block_00032 o13)
        (ori_of block_00034 o6)
        (ori_of block_00004 o21)
        (ori_of block_00002 o14)
        (ori_of block_orange o12)
        (ori_of block_red1 o8)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00020 workspace_table)
        (on block_00030 block_00020)
        (on block_00021 block_00030)
        (clear block_00021)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
