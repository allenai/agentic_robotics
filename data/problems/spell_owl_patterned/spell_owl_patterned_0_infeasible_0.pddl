(define (problem spell_owl_patterned_0_infeasible_0)
    (:domain blocks-tamp)
    (:objects
    block_p block_w block_l block_i block_a block_q - block_letter
    block_orange - block_orange
    block_yellow - block_yellow
    block_green - block_green
    block_red1 - block_red
    workspace_table - table
    robot - gripper    )
    (:init
        (on block_p block_red1)
        (on block_w block_q)
        (on block_l block_a)
        (on block_i block_green)
        (on_table block_a workspace_table)
        (on_table block_q workspace_table)
        (on block_orange block_yellow)
        (on_table block_yellow workspace_table)
        (on_table block_green workspace_table)
        (on_table block_red1 workspace_table)
        (clear block_p)
        (clear block_w)
        (clear block_l)
        (clear block_i)
        (clear block_orange)
        (ori_of block_p o2)
        (ori_of block_w o13)
        (ori_of block_l o13)
        (ori_of block_i o19)
        (ori_of block_a o9)
        (ori_of block_q o1)
        (ori_of block_orange o8)
        (ori_of block_yellow o15)
        (ori_of block_green o24)
        (ori_of block_red1 o12)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (on_table block_l workspace_table)
        (on block_w block_l)
        (on block_p block_w)
        (clear block_p)
    ))

    ;block_size_range = [0.035, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
)
