"""
random agents (hard coded)
"""
from src.agent import Agent
import numpy as np


class Randy(Agent):
    """
    uniformly randomly picks a valid move
    """

    def __init__(self):
        super().__init__()

    def pick_move(self, tower):
        removes, places = tower.valid_moves_product()

        return (removes[np.random.choice(range(len(removes)))],
                places[np.random.choice(range(len(places)))])


class SmartRandy(Randy):
    """
    randomly picks a valid stable move
    imagines placing a block perfectly
    """

    def __init__(self):
        super().__init__()

    def pick_move(self, tower):
        removes, places = tower.valid_moves_product()
        possible = []
        for remove in removes:
            removed = tower.remove_block(remove)
            if not removed.falls():
                for place in places:
                    placed = removed.place_block(place, blk_pos_std=0, blk_angle_std=0)
                    if not placed.falls():
                        possible.append((remove, place))
        if possible:
            return possible[np.random.choice(range(len(possible)))]
        else:
            return super().pick_move(tower=tower)
