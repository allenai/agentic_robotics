(define (problem spell_moa_unpatterned_2_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00001 block_00011 block_00021 block_00027 block_00030 block_00033 block_00022 block_00026 - block_letter
    block_purple - block_purple
    block_green - block_green
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_00001 workspace_table)
        (on_table block_00011 workspace_table)
        (on block_00021 block_purple)
        (on block_00027 block_00001)
        (on_table block_00030 workspace_table)
        (on_table block_00033 workspace_table)
        (on block_00022 block_green)
        (on block_00026 block_00030)
        (on_table block_purple workspace_table)
        (on_table block_green workspace_table)
        (clear block_00011)
        (clear block_00021)
        (clear block_00027)
        (clear block_00033)
        (clear block_00022)
        (clear block_00026)
        (ori_of block_00001 o20)
        (ori_of block_00011 o3)
        (ori_of block_00021 o5)
        (ori_of block_00027 o19)
        (ori_of block_00030 o2)
        (ori_of block_00033 o24)
        (ori_of block_00022 o10)
        (ori_of block_00026 o13)
        (ori_of block_purple o1)
        (ori_of block_green o1)
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
