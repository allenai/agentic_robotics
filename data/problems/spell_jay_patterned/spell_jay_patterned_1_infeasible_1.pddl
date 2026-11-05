(define (problem spell_jay_patterned_1_infeasible_1)
    (:domain blocks-tamp)
    (:objects
    block_j block_x block_y block_f block_o block_p block_i - block_letter
    block_green - block_green
    block_blue - block_blue
    block_red1 - block_red
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_j block_red1)
        (on block_x block_p)
        (on block_y block_o)
        (on block_f block_i)
        (on_table block_o workspace_table)
        (on_table block_p workspace_table)
        (on_table block_i workspace_table)
        (on_table block_green workspace_table)
        (on block_blue block_green)
        (on_table block_red1 workspace_table)
        (clear block_j)
        (clear block_x)
        (clear block_y)
        (clear block_f)
        (clear block_blue)
        (ori_of block_j o11)
        (ori_of block_x o6)
        (ori_of block_y o11)
        (ori_of block_f o10)
        (ori_of block_o o7)
        (ori_of block_p o9)
        (ori_of block_i o3)
        (ori_of block_green o15)
        (ori_of block_blue o5)
        (ori_of block_red1 o9)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_y workspace_table)
        (on block_x block_y)
        (on block_j block_x)
        (clear block_j)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
