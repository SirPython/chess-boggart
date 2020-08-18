import re

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

if __name__ == "__main__":
    with open("lichess_db_standard_rated_2013-01.pgn", "r") as f:
        games = []

        save = False
        while True:
            line = f.readline()

            if line == "": # EOF
                break

            if line[:22] == "[Black \"Desmond_Wilson" or line[:22] == "[White \"Desmond_Wilson":
                save = True
            elif line[0] == "1" and save:
                games.append(line)
                save = False

        # Matches the numbers denoting the start of a new turn, and the ending
        # game result
        ptrn = re.compile(" ?\d+\. | 0-1| 1-0| 1/2-1/2")
        games = tuple(tuple(turn.split(" ") for turn in ptrn.split(game))[1:-1] for game in games)

        print(games[0])