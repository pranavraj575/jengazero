"""
methods to have agents play jenga
defines a generic Agent class that other agents should inherit
"""
from src.tower import *


class Agent:
    def __init__(self):
        pass

    def pick_move(self, tower: Tower):
        raise NotImplementedError


def outcome(agents: [Agent], tower: Tower = None, keep_history=True):
    """
    plays through a game and returns winning player (index)
    :param agents: list of Agents that play the tame
    :param tower: initial tower, if None, initalizes an default leveled tower with default randomness
    :param keep_history: whether to return a history of (tower,action,next_tower,result) tuples played
        result is -1 if the action caused tower to fall
            1 if the game was won (i.e. next player has no moves)
            0 otherwise
    :return: (index of losing player, length of game, (towers played through or None))
    """
    if tower is None:
        tower = Tower()

    turn = 0

    if keep_history:
        history = []
    else:
        history = None
    if tower.falls():
        print("WARNING: tower input was unstable")
        return 0, turn, history
    while tower.has_valid_moves():
        agent = agents[turn%len(agents)]
        remove, place = agent.pick_move(tower)
        new_tower, fell = tower.play_move(remove=remove, place=place)
        result = 0
        if fell:
            result = -1
        if not new_tower.has_valid_moves():
            result = 1

        if keep_history:
            history.append((tower, (remove, place), new_tower, result))
        tower = new_tower
        if fell:
            break
        turn += 1
    # return index of last agent that made a move, number of moves made, tower history
    return turn%len(agents), turn + 1, history


if __name__ == "__main__":
    from agents.randy import Randy, SmartRandy
    from agents.determined import FastPick

    lower, length, hist = (outcome([FastPick(), Randy()], keep_history=True))
    print('loser', lower)
    print('length', length)
    print('last_tower', hist[-1])
    for tower, action, next, terminal in hist:
        print(tower, action, next, terminal)
