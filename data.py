from math import floor

import numpy as np
import chess

def encode_input(fen):
    #tensor = np.zeros((8,8,12), dtype=np.int8)
    tensor = np.zeros(8 * 8 * 12, dtype=np.int8)
    coord = [0, 0]
    pieces = "KQRBNPkqrbnp"

    for rank in fen.split(" ")[0].split("/"):
        for square in rank:
            if square.isdigit():
                coord[1] += int(square)
            else:
                #tensor[coord[0]][coord[1]][pieces.index(square)] = 1
                tensor[(coord[0] * 8) + (coord[1] * 8) + pieces.index(square)] = 1

                coord[1] += 1

        coord[0] += 1
        coord[1] = 0

    return tensor


def encode_output(board, move):
    tensor = np.zeros(64 * 64, dtype=np.int8)

    tensor[ (move.from_square * 64) + move.to_square] = 1
    return tensor

def decode_output(tensor, board):
    while True:
        move_id = np.argmax(tensor)

        from_square = floor(move_id / 64)
        to_square = move_id - (from_square * 64)

        move = chess.Move(
            from_square,
            to_square,
            drop=board.piece_at(to_square) or None
        )

        if move in board.legal_moves:
            return move
        else:
            tensor[move_id] = 0