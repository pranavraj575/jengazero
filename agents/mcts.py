import random
import math
from src.tower import Tower

class Node:
    def __init__(self, state, exploration_constant=math.sqrt(2), parent=None):
        self.state = state
        self.exploration_constant = exploration_constant
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0.0

    def is_fully_expanded(self):
        """
        Check if all possible child states have been explored.

        Returns:
            bool: True if fully expanded, False otherwise.
        """
        return len(self.children) == len(self.state.num_legal_moves)

    def add_child(self, child_state):
        """
        Add a child node to this node.

        Args:
            child_state: The state associated with the new child node.

        Returns:
            Node: The newly created child node.
        """
        child = Node(child_state, self.exploration_constant, self)
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
            exploitation_term = child.score / child.visits
            exploration_term = math.sqrt(2 * math.log(self.visits) / child.visits)
            score = exploitation_term + self.exploration_constant * exploration_term

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

        if self.parent:
            self.parent.backpropagate(score)

def mcts_search(root_state, iterations, exploration_constant=math.sqrt(2), exploitation_constant=1.0):
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
    root_node = Node(root_state, exploration_constant)
    for _ in range(iterations):
        node = root_node
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                next_move = random.choice(node.state.get_possible_moves())
                node = node.add_child(node.state.make_move(next_move))
            else:
                node = node.select_child()

        simulation_result = node.state.evaluate()
        node.backpropagate(simulation_result)

    best_child = max(root_node.children, key=lambda x: x.visits)
    return best_child.state.last_move

class State:
    def __init__(self):
        self.tower = Tower(pos_std=0.001, angle_std=0.001)
        self.valid_removals = None
        self.valid_placements = None

    def get_possible_moves(self):
        if self.valid_removals is None and self.valid_placements is None:
            self.valid_removals, self.valid_placements = self.get_possible_moves()
        return (self.valid_removals, self.valid_placements)

    def make_move(self, move):
        return self.tower.play_move(*move)

    def is_terminal(self):
        return self.tower.terminal_state()

    def evaluate(self):
        return 0.0

# Example usage with a custom State class
initial_state = State()
next_state = mcts_search(initial_state, 1000, math.sqrt(2))
