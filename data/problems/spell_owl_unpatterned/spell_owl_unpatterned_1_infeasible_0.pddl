(define (problem spell_owl_unpatterned_1_infeasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00001 block_00008 block_00005 block_00030 block_00031 block_00035 block_00021 - block_letter
    block_blue - block_blue
    block_green - block_green
    block_red1 - block_red
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_00001 workspace_table)
        (on block_00008 block_green)
        (on block_00005 block_00030)
        (on_table block_00030 workspace_table)
        (on block_00031 block_00021)
        (on block_00035 block_00001)
        (on_table block_00021 workspace_table)
        (on_table block_blue workspace_table)
        (on_table block_green workspace_table)
        (on block_red1 block_blue)
        (clear block_00008)
        (clear block_00005)
        (clear block_00031)
        (clear block_00035)
        (clear block_red1)
        (ori_of block_00001 o10)
        (ori_of block_00008 o16)
        (ori_of block_00005 o16)
        (ori_of block_00030 o6)
        (ori_of block_00031 o20)
        (ori_of block_00035 o6)
        (ori_of block_00021 o19)
        (ori_of block_blue o13)
        (ori_of block_green o15)
        (ori_of block_red1 o7)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00035 workspace_table)
        (on block_00031 block_00035)
        (on block_00030 block_00031)
        (clear block_00030)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
