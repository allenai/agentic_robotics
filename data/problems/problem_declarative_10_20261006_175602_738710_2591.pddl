(define (problem declarative_10_20261006_175602_738710_2591)
    (:domain blocks-tamp)
    (:objects
    block_w block_q block_u block_x block_c block_l block_z - block_letter
    block_red1 - block_red
    block_green - block_green
    block_blue - block_blue
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_w workspace_table)
        (on_table block_red1 workspace_table)
        (on_table block_green workspace_table)
        (on_table block_q workspace_table)
        (on block_u block_w)
        (on_table block_x workspace_table)
        (on_table block_c workspace_table)
        (on block_l block_x)
        (on_table block_blue workspace_table)
        (on block_z block_q)
        (clear block_red1)
        (clear block_green)
        (clear block_u)
        (clear block_c)
        (clear block_l)
        (clear block_blue)
        (clear block_z)
        (ori_of block_w o8)
        (ori_of block_red1 o19)
        (ori_of block_green o3)
        (ori_of block_q o3)
        (ori_of block_u o24)
        (ori_of block_x o16)
        (ori_of block_c o3)
        (ori_of block_l o18)
        (ori_of block_blue o5)
        (ori_of block_z o5)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_w workspace_table)
        (on block_x block_w)
        (on block_green block_x)
        (clear block_green)
        (ori_of block_w o2)
        (ori_of block_x o2)
    ))

    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)