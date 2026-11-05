(define (problem spell_hen_unpatterned_1_infeasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00002 block_00024 block_00013 block_00015 block_00020 block_00003 block_00014 - block_letter
    block_yellow - block_yellow
    block_blue - block_blue
    block_red1 - block_red
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_00002 block_blue)
        (on_table block_00024 workspace_table)
        (on_table block_00013 workspace_table)
        (on_table block_00015 workspace_table)
        (on_table block_00020 workspace_table)
        (on_table block_00003 workspace_table)
        (on_table block_00014 workspace_table)
        (on block_yellow block_00024)
        (on_table block_blue workspace_table)
        (on block_red1 block_00020)
        (clear block_00002)
        (clear block_00013)
        (clear block_00015)
        (clear block_00003)
        (clear block_00014)
        (clear block_yellow)
        (clear block_red1)
        (ori_of block_00002 o14)
        (ori_of block_00024 o13)
        (ori_of block_00013 o10)
        (ori_of block_00015 o2)
        (ori_of block_00020 o14)
        (ori_of block_00003 o2)
        (ori_of block_00014 o22)
        (ori_of block_yellow o14)
        (ori_of block_blue o11)
        (ori_of block_red1 o24)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00020 workspace_table)
        (on block_00003 block_00020)
        (on block_00013 block_00003)
        (clear block_00013)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
