(define (problem spell_owl_patterned_2_feasible_1)
    (:domain blocks-tamp)
    (:objects
    block_o block_w block_l block_e block_g block_h block_a block_m - block_letter
    block_red1 - block_red
    block_blue - block_blue
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_o block_w)
        (on_table block_w workspace_table)
        (on_table block_l workspace_table)
        (on block_e block_h)
        (on block_g block_m)
        (on_table block_h workspace_table)
        (on_table block_a workspace_table)
        (on_table block_m workspace_table)
        (on block_red1 block_l)
        (on block_blue block_a)
        (clear block_o)
        (clear block_e)
        (clear block_g)
        (clear block_red1)
        (clear block_blue)
        (ori_of block_o o9)
        (ori_of block_w o15)
        (ori_of block_l o7)
        (ori_of block_e o2)
        (ori_of block_g o9)
        (ori_of block_h o1)
        (ori_of block_a o9)
        (ori_of block_m o13)
        (ori_of block_red1 o10)
        (ori_of block_blue o1)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_l workspace_table)
        (on block_w block_l)
        (on block_o block_w)
        (clear block_o)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
