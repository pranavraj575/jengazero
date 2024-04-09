"""
deterministic agents (hard coded)
"""
from src.agent import Agent


class FastPick(Agent):
    """
    picks first valid stable move
    otherwise, picks a move
    imagines placing a block perfectly
    """

    def __init__(self):
        super().__init__()

    def pick_move(self, tower):
        removes, places = tower.valid_moves_product()
        for remove in removes:
            removed = tower.remove_block(remove)
            if not removed.deterministic_falls():
                for place in places:
                    placed = removed.place_block(place, blk_pos_std=0, blk_angle_std=0)
                    if not placed.deterministic_falls():
                        return (remove, place)
        return removes[-1], places[-1]
