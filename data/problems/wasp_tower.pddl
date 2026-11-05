(define (problem wasp_tower)
    (:domain blocks-tamp)
    ; Ten colour blocks: two yellow and one black, so an alternating three-tall tower has to put
    ; black in the middle. The other seven are there to be reasoned past. The goal below is only
    ; what Stage-1 codegen consumes; the task the agent is given arrives as its prompt, and no
    ; PDDL goal can say "alternating".
    (:objects
    block_yellow block_yellow2 block_black
    block_red1 block_red2 block_green block_green2 block_blue block_blue2 block_gray - block
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_yellow workspace_table)
        (on_table block_yellow2 workspace_table)
        (on_table block_black workspace_table)
        (on_table block_red1 workspace_table)
        (on_table block_red2 workspace_table)
        (on_table block_green workspace_table)
        (on_table block_green2 workspace_table)
        (on_table block_blue workspace_table)
        (on_table block_blue2 workspace_table)
        (on_table block_gray workspace_table)

        (clear block_yellow)
        (clear block_yellow2)
        (clear block_black)
        (clear block_red1)
        (clear block_red2)
        (clear block_green)
        (clear block_green2)
        (clear block_blue)
        (clear block_blue2)
        (clear block_gray)

        (handempty robot)

        (ori_of block_yellow o1)
        (ori_of block_yellow2 o1)
        (ori_of block_black o1)
        (ori_of block_red1 o1)
        (ori_of block_red2 o1)
        (ori_of block_green o1)
        (ori_of block_green2 o1)
        (ori_of block_blue o1)
        (ori_of block_blue2 o1)
        (ori_of block_gray o1)

        (gripper_at_init)
    )
    (:goal (and
        (on block_black block_yellow)
    ))
)
