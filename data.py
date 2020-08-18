def create_input_tensor(fen):
    tensor = [[[0] * 13] * 8] * 8
    rank = 0
    file = 0
    pieces = "KQRBNPkqrbnp"

    for rank in fen.split(" ")[0].split("/"):
        for square in rank:
            if square.isdigit():
                file += int(square)

            tensor[rank][file][pieces.index(square)] = 1

    return tensor


"""
Literally every possible move ever. This is gonna be really dirty for the first
version

So for every piece except pawn, that's every square (ignore bishop problems)
5 pieces * 64 squares * 2 for capturing/moving = 640

Pawns can move on 64 - 8 = 56 squares. However, for 8 of those moves, they can
promote to 5 possible pieces, turning those 8 moves into 40 moves

So 8 pawns * (64 - 8 - 8 = 48) squares * 8 * 5 = 15360 moves

There's gotta be a better way to do this. I think encoding UCI moves would
be better... one-hot encoding. Encoding issue: bot may try to do moves
that aren't there, although I suppose that's an issue both ways actually

Oooohhhh boy this is gonna be UGGGLLLYYY

Okay so each piece (ignoring pawns right now) has 128 pieces of realestate in
the output vector. Each grouping of two corresponds to a square on the board.
The first move of each grouping is just a move, the second is capturing.

Pawns. Oh boy pawns. We're gonna say each pawn can only move 3/4 of the board,
so that's 48 squares. Let's take away the back rank, because that'll be different.
So now we're at 36 * 2. For the back rank, there are 8 moves, 2 types of moves
(capturing and moving), and then 5 possible promotions. That's 8 * 2 * 5.

So 36 * 2 + 8 * 2 * 5
"""
def create_output_tensor(board, move):
    tensor = [None] * (640 + 1406)

    mover = board.piece_type_at(move.from_square)
    index = (mover - 2) * 128

    # Pawns
    if mover == 1:
        index = 5 * 128 # Past all others' realestate

        index += move.to_square * 2 * 6
        index += 6 if move.drop_pice is not None else 0
        index += move.promotion or 0

    index += move.to_square * 2
    index += 1 if move.drop_piece is not None else 0

    tensor[index] = 1

    return tensor