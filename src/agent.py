"""
generic agent class to play jenga
"""
from src.tower import Tower


class Agent:
    def __init__(self):
        pass

    def pick_move(self, tower: Tower):
        raise NotImplementedError
