"""
methods to have agents play jenga
defines a generic Agent class that other agents should inherit
"""
from typing import List
from src.tower import *


class Agent:
    def __init__(self):
        pass

    def pick_move(self, tower: Tower):
        """
        returns a move ((Layer, index_remove), index_place)
        :param tower: the tower to evalutate
        """
        raise NotImplementedError

    def test_against(self, agent, N=100):
        """
        plays N games against agent, returns the success rate
        """
        lost = 0
        for _ in range(N):
            if np.random.random() < .5:
                agents = [self, agent]
                idx = 0
            else:
                agents = [agent, self]
                idx = 1
            loser, _, hist = outcome(agents)
            if loser == idx:
                lost += 1

        # return the number of times we did not lose
        return 1 - lost/N


def outcome(agents: List[Agent], tower: Tower = None, keep_history=True):
    """
    plays through a game and returns winning player (index)
    :param agents: list of Agents that play the tame
    :param tower: initial tower, if None, initalizes an default leveled tower with default randomness
    :param keep_history: whether to return a history of (tower,action,result) tuples played
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
    if tower.deterministic_falls():
        print("WARNING: tower input was unstable")
        return 0, turn, history
    while tower.has_valid_moves():
        agent = agents[turn%len(agents)]
        remove, place = agent.pick_move(tower)
        new_tower, log_prob_stable = tower.play_move_log_probabilistic(remove=remove, place=place)
        fell=np.random.random()>np.exp(log_prob_stable)
        result = 0
        if fell:
            result = -1
        elif not new_tower.has_valid_moves():
            result = 1

        if keep_history:
            history.append((tower, (remove, place), result))
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
    for tower, action, terminal in hist:
        print(tower, action, terminal)
