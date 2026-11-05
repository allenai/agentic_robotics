(define (problem spell_owl_unpatterned_2_infeasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00001 block_00013 block_00010 block_00030 block_00032 block_00021 block_00015 block_00009 - block_letter
    block_red1 - block_red
    block_blue - block_blue
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_00001 block_00013)
        (on_table block_00013 workspace_table)
        (on_table block_00010 workspace_table)
        (on block_00030 block_00021)
        (on block_00032 block_00009)
        (on_table block_00021 workspace_table)
        (on_table block_00015 workspace_table)
        (on_table block_00009 workspace_table)
        (on block_red1 block_00010)
        (on block_blue block_00015)
        (clear block_00001)
        (clear block_00030)
        (clear block_00032)
        (clear block_red1)
        (clear block_blue)
        (ori_of block_00001 o24)
        (ori_of block_00013 o6)
        (ori_of block_00010 o16)
        (ori_of block_00030 o19)
        (ori_of block_00032 o15)
        (ori_of block_00021 o7)
        (ori_of block_00015 o12)
        (ori_of block_00009 o5)
        (ori_of block_red1 o5)
        (ori_of block_blue o24)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00021 workspace_table)
        (on block_00032 block_00021)
        (on block_00030 block_00032)
        (clear block_00030)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
