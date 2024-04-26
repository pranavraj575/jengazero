import random
import math
import sys
import os

# Add the parent directory to sys.path so that 'src' can be accessed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.agent import Agent, outcome
from src.tower import Tower


class State:
    def __init__(self, tower=None, parent=None, last_move=None, log_stable_prob=0.0):
        if tower is None:
            self.tower = Tower(pos_std=0.001, angle_std=0.001)
        else:
            self.tower = tower
        self.parent = parent
        self.moves = self.tower.valid_moves()
        self.num_legal_moves = len(self.moves)
        self.last_move = last_move
        self.log_stable_prob = log_stable_prob

    def make_move(self, move):
        new_tower, log_stable_prob = self.tower.play_move_log_probabilistic(move[0], move[1])
        return State(tower=new_tower, parent=self, last_move=move, log_stable_prob=log_stable_prob)

    def evaluate(self, fell=False):
        """
        this evaluation measures how good this state is to END in
            i.e. if player i plays a move that leads to this state,
                return 1 if player i will win and -1 if lost
            if this tower has no valid moves, the player that moves to this tower will win (as next player has no moves)
            if this tower falls, the player that moves to this tower will lose
        """  # this will be inverted up the tree, -1 reward for a loss, 1 reward for opp loss
        if fell:
            # this is if the tower fell
            return -1
        if self.num_legal_moves == 0:
            # this is if the tower has no moves left
            return 1
        return 0


class Node:
    def __init__(self, state, exploration_constant=math.sqrt(2), parent=None):
        self.state = state
        self.exploration_constant = exploration_constant
        self.parent = parent
        self.children = []
        self.last_child_idx = 0
        self.visits = 0
        self.score = 0.0

    def is_fully_expanded(self):
        """
        Check if all possible child states have been explored.

        Returns:
            bool: True if fully expanded, False otherwise.
        """
        return len(self.children) == self.state.num_legal_moves

    def add_child(self, child_state):
        """
        Add a child node to this node.

        Args:
            child_state: The state associated with the new child node.

        Returns:
            Node: The newly created child node.
        """
        child = Node(child_state, exploration_constant=self.exploration_constant, parent=self)
        self.children.append(child)
        return child

    def select_child(self):
        """
        Select a child node based on the UCB1 formula.

        Returns:
            Node: The selected child node.
        """
        best_child = None
        best_score = -math.inf

        for child in self.children:
            exploitation_term = child.score/child.visits
            exploration_term = math.sqrt(math.log(self.visits)/child.visits)
            score = exploitation_term + self.exploration_constant*exploration_term
            if score > best_score:
                best_child = child
                best_score = score

        return best_child

    def backpropagate(self, score):
        """
        Backpropagate the simulation result up the tree.

        Args:
            score: The score obtained from the simulation.
        """
        self.visits += 1
        self.score += score

        if self.parent is not None:
            self.parent.backpropagate(-score)


def mcts_search(root_state, iterations, exploration_constant=2*math.sqrt(2)):
    """
    Perform Monte Carlo Tree Search (MCTS) on the given root state to find the best action.

    Args:
        root_state: The initial state of the problem or game.
        iterations: The number of iterations to run the search.
        exploration_constant: The exploration constant (default: sqrt(2)).
        exploitation_constant: The exploitation constant (default: 1.0).

    Returns:
        The best action to take based on the MCTS algorithm.
    """
    # print('searching')
    root_node = Node(root_state, exploration_constant)
    for i in range(iterations):
        # if (i+1) % 100 == 0:
        #     print(f'iteration {i+1}')
        node = root_node
        termination = None
        while True:
            if random.random() <= math.exp(node.state.log_stable_prob):
                termination = 'fell'
                break
            if node.state.num_legal_moves == 0:
                termination = 'no moves'
                break
            if not node.is_fully_expanded():
                next_move = node.state.moves[node.last_child_idx]
                node.last_child_idx += 1
                node = node.add_child(node.state.make_move(next_move))
            else:
                node = node.select_child()
                # print(f'{node.state.tower}\tlog stable prob = {node.state.log_stable_prob:.4f}')
        simulation_result = node.state.evaluate(fell=(termination == 'fell'))
        node.backpropagate(simulation_result)

    best_child = max(root_node.children, key=lambda x: x.score)  # TODO: should this be x.score or x.visits
    return best_child.state.last_move, root_node


class MCTS_player(Agent):
    def __init__(self, num_iterations=1000):
        super().__init__()
        self.num_iterations = num_iterations

    def pick_move(self, tower: Tower):
        state = State(tower)
        best_move, _ = mcts_search(state, self.num_iterations)
        return best_move


# Example usage with a custom State class
if __name__ == "__main__":
    state = State()
    player = 0
    is_terminal = False
    print(f"player {player}'s turn\t{state.tower} log_stable_prob={state.log_stable_prob:.4f}")
    while random.random() <= math.exp(state.log_stable_prob):
        if state.num_legal_moves == 0:
            is_terminal = True
            break
        next_move, node = mcts_search(state, 1000, 2*math.sqrt(2))
        state = state.make_move(next_move)
        player = 1 - player
        for child in node.children:
            exploitation_term = child.score/child.visits
            exploration_term = node.exploration_constant*math.sqrt(math.log(node.visits)/child.visits)
            score = exploitation_term + exploration_term
            print(
                f"{child.state.tower}\texploitation-term={exploitation_term:.4f} \texploration-term={exploration_term:.4f}\tscore={score:.4f}")
        print(f"player {player}'s turn\t{state.tower} log_stable_prob={state.log_stable_prob:.4f}")
    if is_terminal:
        print(f'player {1 - player} won!')
    else:
        print(f'player {player} won!')
