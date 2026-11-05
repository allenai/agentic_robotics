(define (problem spell_hen_patterned_2_infeasible_0)
    (:domain blocks-tamp)
    (:objects
    block_h block_e block_i block_a block_o block_b block_p block_c - block_letter
    block_blue - block_blue
    block_purple - block_purple
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_h workspace_table)
        (on_table block_e workspace_table)
        (on_table block_i workspace_table)
        (on_table block_a workspace_table)
        (on block_o block_purple)
        (on_table block_b workspace_table)
        (on block_p block_a)
        (on block_c block_blue)
        (on_table block_blue workspace_table)
        (on_table block_purple workspace_table)
        (clear block_h)
        (clear block_e)
        (clear block_i)
        (clear block_o)
        (clear block_b)
        (clear block_p)
        (clear block_c)
        (ori_of block_h o11)
        (ori_of block_e o11)
        (ori_of block_i o17)
        (ori_of block_a o23)
        (ori_of block_o o13)
        (ori_of block_b o19)
        (ori_of block_p o9)
        (ori_of block_c o21)
        (ori_of block_blue o13)
        (ori_of block_purple o13)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_i workspace_table)
        (on block_e block_i)
        (on block_h block_e)
        (clear block_h)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
