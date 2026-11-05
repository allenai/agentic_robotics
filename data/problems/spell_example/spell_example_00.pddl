(define (problem spell_example_00)
    (:domain blocks-tamp)
    (:objects
    block_d block_u block_c block_k block_l block_h block_n - block_letter
    block_red1 - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_d workspace_table)
        (clear block_d)
        (on_table block_u workspace_table)
        (on_table block_c workspace_table)
        (on_table block_k workspace_table)
        (on block_l block_k)
        (clear block_l)
        (on block_h block_u)
        (clear block_h)
        (on_table block_n workspace_table)
        (clear block_n)
        (on block_red1 block_c)
        (clear block_red1)
        (ori_of block_d o11)
        (ori_of block_u o21)
        (ori_of block_c o23)
        (ori_of block_k o23)
        (ori_of block_l o9)
        (ori_of block_h o6)
        (ori_of block_n o24)
        (ori_of block_red1 o6)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (ori_of block_d o2)
        (ori_of block_u o2)
        (ori_of block_c o2)
    ))

    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [0, 0.04]
    ;friction_range = [0.1, 0.4]
)