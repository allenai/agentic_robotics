(define (problem spell_owl_unpatterned_0_infeasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00012 block_00008 block_00010 block_00015 block_00032 block_00035 - block_letter
    block_orange - block_orange
    block_yellow - block_yellow
    block_green - block_green
    block_red1 - block_red
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_00012 block_red1)
        (on block_00008 block_00035)
        (on block_00010 block_00032)
        (on block_00015 block_green)
        (on_table block_00032 workspace_table)
        (on_table block_00035 workspace_table)
        (on block_orange block_yellow)
        (on_table block_yellow workspace_table)
        (on_table block_green workspace_table)
        (on_table block_red1 workspace_table)
        (clear block_00012)
        (clear block_00008)
        (clear block_00010)
        (clear block_00015)
        (clear block_orange)
        (ori_of block_00012 o14)
        (ori_of block_00008 o3)
        (ori_of block_00010 o22)
        (ori_of block_00015 o5)
        (ori_of block_00032 o23)
        (ori_of block_00035 o24)
        (ori_of block_orange o23)
        (ori_of block_yellow o3)
        (ori_of block_green o21)
        (ori_of block_red1 o24)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00035 workspace_table)
        (on block_00032 block_00035)
        (on block_00012 block_00032)
        (clear block_00012)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
