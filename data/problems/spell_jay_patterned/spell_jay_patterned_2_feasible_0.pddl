(define (problem spell_jay_patterned_2_feasible_0)
    (:domain blocks-tamp)
    (:objects
    block_j block_a block_y block_f block_c block_r block_i block_n - block_letter
    block_red1 - block_red
    block_orange - block_orange
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_j block_red1)
        (on_table block_a workspace_table)
        (on block_y block_a)
        (on block_f block_r)
        (on_table block_c workspace_table)
        (on_table block_r workspace_table)
        (on_table block_i workspace_table)
        (on block_n block_c)
        (on_table block_red1 workspace_table)
        (on block_orange block_i)
        (clear block_j)
        (clear block_y)
        (clear block_f)
        (clear block_n)
        (clear block_orange)
        (ori_of block_j o9)
        (ori_of block_a o3)
        (ori_of block_y o5)
        (ori_of block_f o7)
        (ori_of block_c o11)
        (ori_of block_r o12)
        (ori_of block_i o10)
        (ori_of block_n o8)
        (ori_of block_red1 o15)
        (ori_of block_orange o14)
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
