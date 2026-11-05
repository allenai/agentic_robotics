(define (problem find_r)
    (:domain blocks-tamp)
    ; Three blocks from the shipped inventory, none of them showing the glyph it is named for:
    ; face five carries that glyph, and `front_face` is five only at o2, o6, o10 and o14. So the
    ; agent has to turn blocks to find out what it has. block_r starts at o8, whose front face is
    ; six -- the opposite face, which on R.usd carries the E it is paired with.
    (:objects
    block_r - block_r_e
    block_o1 - block_o_b
    block_t - block_t_g
    workspace_table - table
    robot - gripper    )
    (:init
        (on_table block_r workspace_table)
        (on_table block_o1 workspace_table)
        (on_table block_t workspace_table)

        (clear block_r)
        (clear block_o1)
        (clear block_t)

        (handempty robot)

        (ori_of block_r o8)
        (ori_of block_o1 o1)
        (ori_of block_t o4)

        (gripper_at_init)
    )
    (:goal (and
        (on_table block_r workspace_table)
        (ori_of block_r o2)
    ))
)
