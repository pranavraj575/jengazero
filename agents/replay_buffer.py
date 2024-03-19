from src.agent import *

from collections import namedtuple, deque
import pickle, random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))


class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def save(self, filename):
        f=open(filename,'wb')
        pickle.dump(self.memory, f)
        f.close()

    def load(self, filename):
        f=open(filename, 'rb')
        self.memory = pickle.load(f)
        f.close()

    def size(self):
        return len(self.memory)

    def __len__(self):
        return len(self.memory)


def add_training_data(replay: ReplayMemory, agent1: Agent, agent2: Agent, tower: Tower = None):
    """
    creates training data by having agents play, and sends it into replay buffer
    """
    loser, _, history = outcome([agent1, agent2], tower=tower)
    for tower, action, next_tower, result in history:
        state = tower.to_array()
        next_state = next_tower.to_array()
        reward = float(result)  # reward is -1 if tower fell, 1 if the game was won on this move, 0 otherwise
        terminal = bool(result)  # terminal is whether either the tower fell or the game was won
        replay.push(state, action, next_state, reward, terminal)


if __name__ == "__main__":
    from agents.determined import FastPick
    from agents.randy import Randy
    import os, sys

    DIR = os.path.abspath(os.path.dirname(os.path.dirname(sys.argv[0])))

    test = ReplayMemory(200)
    add_training_data(test, Randy(), FastPick())
    state, action, next_state, reward, terminal = (test.memory[-1])
    print(tower_from_array(state), action, tower_from_array(next_state), reward, terminal)
    test_file = os.path.join(DIR, 'replay_test.pkl')
    test.save(test_file)

    test2 = ReplayMemory(69)
    print(test2.memory.maxlen)
    test2.load(test_file)
    print(test2.memory.maxlen)
    state, action, next_state, reward, terminal = (test2.memory[-1])
    print(tower_from_array(state), action, tower_from_array(next_state), reward, terminal)
    os.remove(test_file)
