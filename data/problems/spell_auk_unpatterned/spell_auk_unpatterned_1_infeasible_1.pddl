(define (problem spell_auk_unpatterned_1_infeasible_1)
    (:domain blocks-tamp)
    (:objects
    block_00020 block_00021 block_00027 block_00026 block_00032 block_00024 block_00006 - block_letter
    block_red1 - block_red
    block_yellow - block_yellow
    block_purple - block_purple
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_00020 workspace_table)
        (on_table block_00021 workspace_table)
        (on_table block_00027 workspace_table)
        (on_table block_00026 workspace_table)
        (on_table block_00032 workspace_table)
        (on_table block_00024 workspace_table)
        (on block_00006 block_red1)
        (on_table block_red1 workspace_table)
        (on_table block_yellow workspace_table)
        (on block_purple block_00027)
        (clear block_00020)
        (clear block_00021)
        (clear block_00026)
        (clear block_00032)
        (clear block_00024)
        (clear block_00006)
        (clear block_yellow)
        (clear block_purple)
        (ori_of block_00020 o15)
        (ori_of block_00021 o4)
        (ori_of block_00027 o21)
        (ori_of block_00026 o17)
        (ori_of block_00032 o21)
        (ori_of block_00024 o19)
        (ori_of block_00006 o16)
        (ori_of block_red1 o17)
        (ori_of block_yellow o7)
        (ori_of block_purple o17)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00020 workspace_table)
        (on block_00026 block_00020)
        (on block_00027 block_00026)
        (clear block_00027)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
