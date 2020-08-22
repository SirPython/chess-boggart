import re
import chess.pgn
import chess
import numpy as np
import csv
from tensorflow.keras import models, layers
import sys
import os

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
    if sys.argv[1] == "load_games":
        with open(sys.argv[2], "r") as pgn:
            while (game := chess.pgn.read_game(pgn)) != None:
                if game.headers["White"] == sys.argv[3] or game.headers["Black"] == sys.argv[3]:
                    print(game, file=open(sys.argv[4], "a"), end="\n\n")
                    # TODO: Save in a python file and let pycache take it
    elif sys.argv[1] == "train_model":
        training_set = []
        labels = []
        games = []

        with open(sys.argv[2], "r") as pgn:
            while (game := chess.pgn.read_game(pgn)) != None:
                games.append(game)

        for game in games:
            board = game.board()

            skip = False if game.headers["White"] == sys.argv[3] else True
            for move in game.mainline_moves():
                # Only consider the positions where our player had to make a move
                if skip:
                    skip = False
                    board.push(move)
                    continue
                skip = True

                training_set.append(data.encode_input(board.fen()))
                labels.append(data.encode_output(board, move))

                board.push(move)

        print(len(training_set))

        #print(len(training_set), len(labels))
        training_set = np.array(training_set)
        labels = np.array(labels)

        model = models.Sequential()
        model.add(layers.Dense(64, activation="relu", input_shape=(8 * 8 * 12,)))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(2048, activation="softmax"))

        model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(training_set, labels, epochs=50, batch_size=16)

        model.predict(training_set[0])

        model.save("model.h5")

    elif sys.argv[1] == "play":
        model = models.load_model(sys.argv[2])

        board = chess.Board()
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(board)

            player_move = input("Enter move:")
            board.push_san(player_move)

            os.system('cls' if os.name == 'nt' else 'clear')
            print(board)

            bot_move = data.encode_input(board.fen()).shape
            print(bot_move)
            bot_move = model.predict(data.encode_input(board.fen()), verbose=1)
            print(bot_move)

            input("hold")


    else:
        print("You need a command buddy")
