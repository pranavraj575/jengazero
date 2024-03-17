"""
methods and such to play the game of jenga
"""
from src.agent import Agent
from src.tower import Tower


def outcome(agents: [Agent], tower: Tower = None, history=True):
    """
    plays through a game and returns winning player (index)
    :param agents: list of Agents that play the tame
    :param tower: initial tower, if None, initalizes an 18 level tower with default randomness
    :param history: whether to return a history of towers
    :return: (index of losing player, length of game, (towers played through or None))
    """
    if tower is None:
        tower = Tower(default_ht=18)

    turn = 0

    if history:
        towers = [tower]
    else:
        towers = None
    if tower.falls():
        print("WARNING: tower input was unstable")
        return 0, turn, towers
    while tower.has_valid_moves():
        agent = agents[turn%len(agents)]
        remove, place = agent.pick_move(tower)
        tower, fell = tower.play_move(remove=remove, place=place)
        if history:
            towers.append(tower)
        if fell:
            break
        turn += 1
    # return index of last agent that made a move, number of moves made, tower history
    return turn%len(agents), turn + 1, towers


if __name__ == "__main__":
    from agents.randy import Randy, SmartRandy
    from agents.determined import FastPick

    lower, length, hist = (outcome([FastPick(), Randy()], history=True))
    print('loser', lower)
    print('length', length)
    print('last_tower', hist[-1])
