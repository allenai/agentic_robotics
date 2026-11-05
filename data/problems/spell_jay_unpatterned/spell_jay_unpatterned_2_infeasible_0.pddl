(define (problem spell_jay_unpatterned_2_infeasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00000 block_00025 block_00012 block_00003 block_00021 block_00027 block_00024 block_00023 - block_letter
    block_red1 - block_red
    block_orange - block_orange
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_00000 block_red1)
        (on_table block_00025 workspace_table)
        (on block_00012 block_00025)
        (on block_00003 block_00027)
        (on_table block_00021 workspace_table)
        (on_table block_00027 workspace_table)
        (on_table block_00024 workspace_table)
        (on block_00023 block_00021)
        (on_table block_red1 workspace_table)
        (on block_orange block_00024)
        (clear block_00000)
        (clear block_00012)
        (clear block_00003)
        (clear block_00023)
        (clear block_orange)
        (ori_of block_00000 o21)
        (ori_of block_00025 o8)
        (ori_of block_00012 o1)
        (ori_of block_00003 o3)
        (ori_of block_00021 o23)
        (ori_of block_00027 o23)
        (ori_of block_00024 o18)
        (ori_of block_00023 o1)
        (ori_of block_red1 o22)
        (ori_of block_orange o10)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00025 workspace_table)
        (on block_00021 block_00025)
        (on block_00000 block_00021)
        (clear block_00000)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
