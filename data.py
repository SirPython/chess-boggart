import numpy as np

def encode_input(fen):
    tensor = [[[0] * 12] * 8] * 8
    tensor = np.zeros((8,8,12), np.int8)
    coord = [0, 0]
    pieces = "KQRBNPkqrbnp"

    for rank in fen.split(" ")[0].split("/"):
        for square in rank:
            if square.isdigit():
                coord[1] += int(square)
            else:
                tensor[coord[0]][coord[1]][pieces.index(square)] = 1

                coord[1] += 1

        coord[0] += 1
        coord[1] = 0

    return tensor


"""
Oooohhhh boy this is gonna be UGGGLLLYYY.

One-hot encoding every single possible move ever.

Okay so each piece (ignoring pawns right now) has 128 pieces of realestate in
the output vector. Each grouping of two corresponds to a square on the board.
The first move of each grouping is just a move, the second is capturing.

Pawns. Oh boy pawns. We're gonna say each pawn can only move 3/4 of the board,
so that's 48 squares. Let's take away the back rank, because that'll be different.
So now we're at 36 * 2. For the back rank, there are 8 moves, 2 types of moves
(capturing and moving), and then 5 possible promotions. That's 8 * 2 * 5.

So 36 * 2 + 8 * 2 * 5
"""
def encode_output(board, move):
    # TODO If this will work with bits intead...
    tensor = np.zeros(2048) # It actually comes out to this. Crazy.

    mover = board.piece_type_at(move.from_square)

    print("mover", mover)
    if mover == 1: # Pawns
        """
        index = (
            5    # 5 pieces besides pawn (they own the first realestate in the encoding)
            * 64 # 64 squares to move
            * 2  # Two types of moves
        )        # Index now points past realestate of non-pawns; the rest is for pawns
        + (
            move.to_square
            * 2  # two types of moves
            * 6  # 4 promotions + 1 for none + 1 because queen is 6, not 5  :()
        )
        + (
            6 if move.drop is not None else 0
        )        # Capture move or not
        + (
            move.promotion if move.promotion else 0
        )
        """
        index = 640 + (move.to_square * 12) + (6 if move.drop is not None else 0) + (move.promotion or 0)
    else: # Every other piece
        """
        index = (
            (mover - 2) # Pawns are 1, so this 0 indexes the pieces starting with knights
            * 64        # 64 squares
            * 2         # two types of move per square
        )               # Index is now pointing to 128 indecies of own real estate
        + (
            move.to_square
            * 2         # Index is now pointing to 2 index group to describe moving or capturing
        )
        + (
            1 if move.drop is not None else 0
        )
        """
        index = ((mover - 2) * 128) + (move.to_square * 2) + (1 if move.drop is not None else 0)

    tensor[index] = 1
    return tensor