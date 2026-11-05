(define (problem spell_yes_01)
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
        (clear block_s)
        (on_table block_k workspace_table)
        (clear block_k)
        (on_table block_c workspace_table)
        (clear block_c)
        (on block_j block_e)
        (clear block_j)
        (on_table block_z workspace_table)
        (clear block_z)
        (on_table block_blue workspace_table)
        (clear block_blue)
        (on_table block_orange workspace_table)
        (clear block_orange)
        (ori_of block_y o11)
        (ori_of block_e o24)
        (ori_of block_s o9)
        (ori_of block_k o23)
        (ori_of block_c o15)
        (ori_of block_j o5)
        (ori_of block_z o12)
        (ori_of block_blue o8)
        (ori_of block_orange o10)
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