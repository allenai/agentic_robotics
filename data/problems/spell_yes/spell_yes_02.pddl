(define (problem spell_yes_02)
    (:domain blocks-tamp)
    (:objects
    block_y block_e block_s block_k block_c block_j block_z - block_letter
    block_blue block_orange - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_y workspace_table)
        (clear block_y)
        (on_table block_e workspace_table)
        (on_table block_s workspace_table)
        (on_table block_k workspace_table)
        (clear block_k)
        (on block_c block_e)
        (clear block_c)
        (on_table block_j workspace_table)
        (clear block_j)
        (on block_z block_s)
        (clear block_z)
        (on_table block_blue workspace_table)
        (clear block_blue)
        (on_table block_orange workspace_table)
        (clear block_orange)
        (ori_of block_y o1)
        (ori_of block_e o19)
        (ori_of block_s o22)
        (ori_of block_k o24)
        (ori_of block_c o21)
        (ori_of block_j o13)
        (ori_of block_z o1)
        (ori_of block_blue o2)
        (ori_of block_orange o24)
        (handempty robot)
        (gripper_at_init)
    )
    (:goal (and
        (ori_of block_y o2)
        (ori_of block_e o2)
        (ori_of block_s o2)
    ))

    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [0, 0.04]
    ;friction_range = [0.1, 0.4]
)