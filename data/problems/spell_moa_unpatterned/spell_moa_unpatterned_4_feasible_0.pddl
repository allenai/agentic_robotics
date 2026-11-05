(define (problem spell_moa_unpatterned_4_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_00001 block_00011 block_00021 block_00027 block_00030 block_00033 block_00017 block_00013 block_00016 block_00022 - block_letter
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_00001 workspace_table)
        (on block_00011 block_00001)
        (on_table block_00021 workspace_table)
        (on_table block_00027 workspace_table)
        (on block_00030 block_00021)
        (on_table block_00033 workspace_table)
        (on block_00017 block_00013)
        (on_table block_00013 workspace_table)
        (on block_00016 block_00033)
        (on_table block_00022 workspace_table)
        (clear block_00011)
        (clear block_00027)
        (clear block_00030)
        (clear block_00017)
        (clear block_00016)
        (clear block_00022)
        (ori_of block_00001 o23)
        (ori_of block_00011 o11)
        (ori_of block_00021 o20)
        (ori_of block_00027 o4)
        (ori_of block_00030 o13)
        (ori_of block_00033 o12)
        (ori_of block_00017 o1)
        (ori_of block_00013 o9)
        (ori_of block_00016 o24)
        (ori_of block_00022 o20)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_00021 workspace_table)
        (on block_00030 block_00021)
        (on block_00033 block_00030)
        (clear block_00033)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
