(define (problem spell_jay_unpatterned_0_feasible_1)
    (:domain blocks-tamp)
    (:objects
    block_00000 block_00001 block_00012 block_00016 block_00021 block_00027 - block_letter
    block_orange - block_orange
    block_blue - block_blue
    block_purple - block_purple
    block_red1 - block_red
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_00000 block_00001)
        (on_table block_00001 workspace_table)
        (on_table block_00012 workspace_table)
        (on block_00016 block_purple)
        (on_table block_00021 workspace_table)
        (on block_00027 block_00021)
        (on block_orange block_00012)
        (on_table block_blue workspace_table)
        (on_table block_purple workspace_table)
        (on block_red1 block_blue)
        (clear block_00000)
        (clear block_00016)
        (clear block_00027)
        (clear block_orange)
        (clear block_red1)
        (ori_of block_00000 o11)
        (ori_of block_00001 o6)
        (ori_of block_00012 o18)
        (ori_of block_00016 o1)
        (ori_of block_00021 o8)
        (ori_of block_00027 o11)
        (ori_of block_orange o20)
        (ori_of block_blue o21)
        (ori_of block_purple o9)
        (ori_of block_red1 o22)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00016 workspace_table)
        (on block_00021 block_00016)
        (on block_00012 block_00021)
        (clear block_00012)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
