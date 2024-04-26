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
        random.shuffle(self.moves)
        self.num_legal_moves = len(self.moves)
        self.last_move = last_move
        self.log_stable_prob = log_stable_prob
        # self.is_terminal = False

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
        """
        # this will be inverted up the tree, -1 reward for a loss, 1 reward for opp loss
        if fell:
            # this is if the tower fell
            return -1
        if self.num_legal_moves == 0:
            # this is if the tower has no moves left
            return 1
        return 0


class Node:
    def __init__(self, state: State, exploration_constant=math.sqrt(2), parent=None):
        self.state = state
        self.exploration_constant = exploration_constant
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
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


def random_playout(root_state: State, depth=float('inf'), trials=1):
    """
    preform a random playout from the root state
        estimates how good root_state is to END at
    assumes root state has legal moves
    takes into account the probability of the root state falling
    Args:
        root_state: initial state
        depth: max depth to search to (default infinite)
        trials: number of trials to take from root (default 1)
    Return:
        in general, runs eval on the termainal state and propegates it back to root state
        takes average if trials>1
    """
    score = 0
    for trial in range(trials):
        if random.random() > math.exp(root_state.log_stable_prob):
            # THIS tower fell
            # then the last player lost
            score += -1
            continue
        # now we consider the next state
        next_move = root_state.moves[random.randint(0, root_state.num_legal_moves - 1)]
        next_state = root_state.make_move(next_move)
        if next_state.num_legal_moves == 0:
            # no legal moves here, losing state for player that ended at root_state
            score += -1
            continue
        if depth <= 0:
            # we hit the depth limit
            goodness = next_state.evaluate(fell=False)
            # this is an estimate of how good the next state is to END at
            # so we invert and add this to score
            score += -goodness
            continue
        # otherwise we now have to evaluate the next state
        next_outcome = random_playout(next_state, depth=depth - 1, trials=1)
        score += -next_outcome
    return score/trials


def mcts_sample(root_node: Node, depth=float('inf'), check_root_stability=False):
    """
    preform a mcts sample from the root node and backpropogates result
        updates measure how good root_node is to END at
    assumes root state is always non-terminal
        root state will not fall over
        root state has legal moves
    Args:
        root_state: initial state
        depth: max depth to search to (default infinite)
    """
    if check_root_stability:
        if random.random() > math.exp(root_node.state.log_stable_prob):
            # if the root immediately falls, it was not a good state to end at
            root_node.backpropagate(-1)
    if depth <= 0:
        fell = random.random() > math.exp(root_node.state.log_stable_prob)
        root_node.state.evaluate(fell=fell)

    if not root_node.is_fully_expanded():
        # we must rollout from a random child, and add that child to the tree
        next_move = root_node.state.moves[root_node.last_child_idx]
        root_node.last_child_idx += 1
        child = root_node.state.make_move(next_move)
        # we will now measure how good a state child is to END in
        if child.num_legal_moves == 0:
            # in this case root_node just won
            outcome = 1
        else:
            # estimate with a random game
            outcome = random_playout(child, depth=depth - 1, trials=1)
        child_node = root_node.add_child(child)
        child_node.backpropagate(outcome)
    else:
        child_node = root_node.select_child()
        if random.random() > math.exp(child_node.state.log_stable_prob):
            child_node.backpropagate(-1)
        elif child_node.state.num_legal_moves == 0:
            child_node.backpropagate(1)
        else:
            mcts_sample(child_node, depth=depth - 1, check_root_stability=False)


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
        mcts_sample(root_node)
        continue
        """
        # if (i+1) % 100 == 0:
        #     print(f'iteration {i+1}')
        node = root_node
        termination = None
        while True:
            if random.random() > math.exp(node.state.log_stable_prob):
                termination = 'fell'
                break
            if node.state.num_legal_moves == 0:
                # node.state.is_terminal = True
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
        # node.state.is_terminal = False
        node.backpropagate(simulation_result)"""
    best_child = max(root_node.children, key=lambda x: x.score) # TODO: should this be x.score or x.visits
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

        print(
            f"{node.state.tower}\texploitation-term={node.score/node.visits:.4f} \t")
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
