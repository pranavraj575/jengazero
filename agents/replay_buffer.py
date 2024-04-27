from src.agent import *
from src.utils import *

from collections import namedtuple, deque
import pickle

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))
State_Reward_Distrib = namedtuple('State_Reward_Distrib',
                        ('state', 'reward', 'distribution'))


class ReplayMemory:
    def __init__(self, capacity=10000,namedtup=Transition):
        self.memory = deque([], maxlen=capacity)
        self.namedtup=namedtup

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.namedtup(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def full(self):
        """
        returns whetehr memory is full
        """
        return len(self.memory)==self.memory.maxlen
    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.memory, f)
        f.close()

    def load(self, filename):
        f = open(filename, 'rb')
        self.memory = pickle.load(f)
        f.close()

    def size(self):
        return len(self.memory)

    def __len__(self):
        return len(self.memory)


def transfer_SRD(buffer:ReplayMemory):
    temp=ReplayMemory(capacity=buffer.memory.maxlen, namedtup=State_Reward_Distrib)
    transferring=type(buffer.memory[0].state)==Tower
    if transferring:
        print("TRANFERRING")
    for thingy in buffer.memory:
        temp.push(thingy.state.to_array() if type(thingy.state)==Tower else thingy.state, thingy.reward, thingy.distribution)
    return temp

def add_training_data(replay: ReplayMemory, agent1: Agent, agent2: Agent, tower: Tower = None,
                      skip_opponent_step=True):
    """
    creates training data by having agents play, and sends it into replay buffer
    :param skip_opponent_step: if true, models the 'next tower' as the players next move
        otherwise, models 'next tower' as opponents tower, and something must be done about this when learning
    """
    loser, _, history = outcome([agent1, agent2], tower=tower)
    for i in range(len(history) - 2):
        tower, action, result = history[i]
        opponent_tower, _, _ = history[i + 1]
        player_next_tower, _, _ = history[i + 2]

        state = tower.to_array()
        if skip_opponent_step:
            next_state = player_next_tower.to_array()
        else:
            next_state = opponent_tower.to_array()
        reward = float(result)  # this should always be 0
        terminal = bool(result)  # this should always be False
        replay.push(state, action, next_state, reward, terminal)

    # now handle last two steps
    if len(history) >= 1:  # this should always be true, as first tower never falls
        losing_tower, losing_action, losing_result = history[-1]
        replay.push(losing_tower.to_array(), losing_action, None, -1., True)
    if len(history) >= 2:  # this only runs if the first player succeeded on the first move
        winning_tower, winning_action, winning_result = history[-2]
        if skip_opponent_step:
            # we model this as a 'success' step, causeing the agent to win the game
            replay.push(winning_tower.to_array(), winning_action, None, 1., True)
        else:
            # otherwise, we just model this as another non-terminal state
            losing_tower, _, _ = history[-1]
            replay.push(winning_tower.to_array(), winning_action, losing_tower.to_array(), 0., False)


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
