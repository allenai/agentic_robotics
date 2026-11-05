(define (problem spell_jay_unpatterned_1_infeasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00000 block_00001 block_00012 block_00016 block_00034 block_00026 block_00011 - block_letter
    block_green - block_green
    block_blue - block_blue
    block_red1 - block_red
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_00000 block_red1)
        (on block_00001 block_00026)
        (on block_00012 block_00034)
        (on block_00016 block_00011)
        (on_table block_00034 workspace_table)
        (on_table block_00026 workspace_table)
        (on_table block_00011 workspace_table)
        (on_table block_green workspace_table)
        (on block_blue block_green)
        (on_table block_red1 workspace_table)
        (clear block_00000)
        (clear block_00001)
        (clear block_00012)
        (clear block_00016)
        (clear block_blue)
        (ori_of block_00000 o5)
        (ori_of block_00001 o1)
        (ori_of block_00012 o17)
        (ori_of block_00016 o2)
        (ori_of block_00034 o4)
        (ori_of block_00026 o22)
        (ori_of block_00011 o5)
        (ori_of block_green o19)
        (ori_of block_blue o16)
        (ori_of block_red1 o10)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00016 workspace_table)
        (on block_00026 block_00016)
        (on block_00012 block_00026)
        (clear block_00012)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
