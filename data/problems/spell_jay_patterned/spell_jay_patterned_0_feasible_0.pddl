(define (problem spell_jay_patterned_0_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_j block_a block_y block_k block_x block_i - block_letter
    block_orange - block_orange
    block_blue - block_blue
    block_purple - block_purple
    block_red1 - block_red
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_j block_a)
        (on_table block_a workspace_table)
        (on_table block_y workspace_table)
        (on block_k block_purple)
        (on_table block_x workspace_table)
        (on block_i block_x)
        (on block_orange block_y)
        (on_table block_blue workspace_table)
        (on_table block_purple workspace_table)
        (on block_red1 block_blue)
        (clear block_j)
        (clear block_k)
        (clear block_i)
        (clear block_orange)
        (clear block_red1)
        (ori_of block_j o1)
        (ori_of block_a o19)
        (ori_of block_y o17)
        (ori_of block_k o6)
        (ori_of block_x o22)
        (ori_of block_i o19)
        (ori_of block_orange o3)
        (ori_of block_blue o6)
        (ori_of block_purple o22)
        (ori_of block_red1 o7)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_y workspace_table)
        (on block_a block_y)
        (on block_j block_a)
        (clear block_j)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
