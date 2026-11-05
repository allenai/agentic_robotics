(define (problem spell_owl_patterned_1_infeasible_0)
    (:domain blocks-tamp)
    (:objects
    block_o block_y block_l block_t block_u block_c block_d - block_letter
    block_blue - block_blue
    block_green - block_green
    block_red1 - block_red
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_o workspace_table)
        (on block_y block_green)
        (on block_l block_t)
        (on_table block_t workspace_table)
        (on block_u block_d)
        (on block_c block_o)
        (on_table block_d workspace_table)
        (on_table block_blue workspace_table)
        (on_table block_green workspace_table)
        (on block_red1 block_blue)
        (clear block_y)
        (clear block_l)
        (clear block_u)
        (clear block_c)
        (clear block_red1)
        (ori_of block_o o11)
        (ori_of block_y o14)
        (ori_of block_l o11)
        (ori_of block_t o10)
        (ori_of block_u o15)
        (ori_of block_c o19)
        (ori_of block_d o3)
        (ori_of block_blue o10)
        (ori_of block_green o9)
        (ori_of block_red1 o11)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_l workspace_table)
        (on block_y block_l)
        (on block_o block_y)
        (clear block_o)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
