(define (problem spell_emu_unpatterned_2_infeasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00011 block_00012 block_00022 block_00032 block_00033 block_00031 block_00005 block_00025 - block_letter
    block_purple - block_purple
    block_green - block_green
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_00011 block_purple)
        (on_table block_00012 workspace_table)
        (on_table block_00022 workspace_table)
        (on_table block_00032 workspace_table)
        (on_table block_00033 workspace_table)
        (on block_00031 block_00033)
        (on block_00005 block_00032)
        (on_table block_00025 workspace_table)
        (on_table block_purple workspace_table)
        (on block_green block_00022)
        (clear block_00011)
        (clear block_00012)
        (clear block_00031)
        (clear block_00005)
        (clear block_00025)
        (clear block_green)
        (ori_of block_00011 o3)
        (ori_of block_00012 o8)
        (ori_of block_00022 o24)
        (ori_of block_00032 o24)
        (ori_of block_00033 o14)
        (ori_of block_00031 o8)
        (ori_of block_00005 o12)
        (ori_of block_00025 o8)
        (ori_of block_purple o3)
        (ori_of block_green o2)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00031 workspace_table)
        (on block_00033 block_00031)
        (on block_00012 block_00033)
        (clear block_00012)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
