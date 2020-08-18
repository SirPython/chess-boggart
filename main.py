import re
import chess

import data

"""
Data processing:
1. Get Lichess username
2. Get N games in which the user won... or maybe all games?
3. In each game, get all the board positions where it was the user's turn
4. Convert each board position into tensor the size: 8x8x13
5. Convert the move the user played into the big vector idea thing
6. Save locally

Ideas:
- Maybe make it learn a player twice: for black and for white
- Perhaps simply the board state is not good enough for input. Maybe
responsibilities, threats, etc. would be better
"""

NAME = "Desmond_Wilson"

if __name__ == "__main__":
    with open("lichess_db_standard_rated_2013-01.pgn", "r") as pgn:
        training_set = []
        labls = []

        games = []

        while True:
            game = chess.pgn.read_game(pgn)

            if game.headers["White"] == NAME or game.headers["Black"] == NAME:
                games.append(game)

        for game in games:
            board = game.board()

            skip = False if game.headers["White"] == Name else True
            for move in game.mainline_moves():
                # Only consider the positions where our player had to make a move
                if skip:
                    skip = False
                    continue

                training_set.append(data.create_input_tensor(board.fen()))
                labels.append(data.create_output_tensor(move))