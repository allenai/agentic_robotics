(define (problem spell_emu_unpatterned_0_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00011 block_00012 block_00030 block_00032 block_00033 block_00034 - block_letter
    block_green - block_green
    block_red1 - block_red
    block_orange - block_orange
    block_yellow - block_yellow
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_00011 workspace_table)
        (on block_00012 block_00032)
        (on block_00030 block_orange)
        (on_table block_00032 workspace_table)
        (on_table block_00033 workspace_table)
        (on block_00034 block_red1)
        (on block_green block_00011)
        (on_table block_red1 workspace_table)
        (on_table block_orange workspace_table)
        (on_table block_yellow workspace_table)
        (clear block_00012)
        (clear block_00030)
        (clear block_00033)
        (clear block_00034)
        (clear block_green)
        (clear block_yellow)
        (ori_of block_00011 o17)
        (ori_of block_00012 o5)
        (ori_of block_00030 o5)
        (ori_of block_00032 o19)
        (ori_of block_00033 o6)
        (ori_of block_00034 o2)
        (ori_of block_green o6)
        (ori_of block_red1 o2)
        (ori_of block_orange o10)
        (ori_of block_yellow o8)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00030 workspace_table)
        (on block_00011 block_00030)
        (on block_00032 block_00011)
        (clear block_00032)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
