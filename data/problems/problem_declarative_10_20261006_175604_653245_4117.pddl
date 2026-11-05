(define (problem declarative_10_20261006_175604_653245_4117)
    (:domain blocks-tamp)
    (:objects
    block_m block_p block_w block_q block_f block_l block_s - block_letter
    block_green - block_green
    block_blue - block_blue
    block_yellow - block_yellow
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_m workspace_table)
        (on_table block_p workspace_table)
        (on_table block_w workspace_table)
        (on_table block_green workspace_table)
        (on_table block_q workspace_table)
        (on_table block_f workspace_table)
        (on block_blue block_w)
        (on_table block_l workspace_table)
        (on_table block_yellow workspace_table)
        (on_table block_s workspace_table)
        (clear block_m)
        (clear block_p)
        (clear block_green)
        (clear block_q)
        (clear block_f)
        (clear block_blue)
        (clear block_l)
        (clear block_yellow)
        (clear block_s)
        (ori_of block_m o3)
        (ori_of block_p o23)
        (ori_of block_w o9)
        (ori_of block_green o10)
        (ori_of block_q o17)
        (ori_of block_f o19)
        (ori_of block_blue o10)
        (ori_of block_l o1)
        (ori_of block_yellow o1)
        (ori_of block_s o19)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_m workspace_table)
        (on block_q block_m)
        (on block_p block_q)
        (clear block_p)
        (ori_of block_m o2)
        (ori_of block_q o2)
        (ori_of block_p o2)
    ))

    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)