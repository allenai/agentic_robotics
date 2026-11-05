(define (problem spell_auk_unpatterned_0_feasible_1)
    (:domain blocks-tamp)
    (:objects
    block_00020 block_00021 block_00027 block_00030 block_00032 block_00034 - block_letter
    block_red1 - block_red
    block_yellow - block_yellow
    block_orange - block_orange
    block_blue - block_blue
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_00020 workspace_table)
        (on_table block_00021 workspace_table)
        (on block_00027 block_00021)
        (on_table block_00030 workspace_table)
        (on block_00032 block_blue)
        (on_table block_00034 workspace_table)
        (on_table block_red1 workspace_table)
        (on_table block_yellow workspace_table)
        (on_table block_orange workspace_table)
        (on_table block_blue workspace_table)
        (clear block_00020)
        (clear block_00027)
        (clear block_00030)
        (clear block_00032)
        (clear block_00034)
        (clear block_red1)
        (clear block_yellow)
        (clear block_orange)
        (ori_of block_00020 o16)
        (ori_of block_00021 o15)
        (ori_of block_00027 o19)
        (ori_of block_00030 o16)
        (ori_of block_00032 o9)
        (ori_of block_00034 o19)
        (ori_of block_red1 o9)
        (ori_of block_yellow o15)
        (ori_of block_orange o7)
        (ori_of block_blue o3)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00032 workspace_table)
        (on block_00034 block_00032)
        (on block_00027 block_00034)
        (clear block_00027)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
