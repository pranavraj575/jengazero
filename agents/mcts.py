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
    """
    what a state must implement
    added this structure so we can change evaluation function for alphazero
    """

    def __init__(self, tower, parent, last_move, log_stable_prob):
        if tower is None:
            self.tower = Tower(pos_std=0.001, angle_std=0.001)
        else:
            self.tower = tower
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.moves = self.tower.valid_moves()
        random.shuffle(self.moves)
        self.num_legal_moves = len(self.moves)
        self.last_move = last_move
        self.log_stable_prob = log_stable_prob

    def make_move(self, move):
        """
        returns state resulting from the move
        """
        raise NotImplementedError

    def evaluate(self, fell):
        """
        returns a scalar evaluating the state
        this evaluation measures how good this state is to END in
        """
        raise NotImplementedError


class BasicState(State):
    def __init__(self, tower=None, parent=None, last_move=None, log_stable_prob=0.0):
        super().__init__(tower=tower,
                         parent=parent,
                         last_move=last_move,
                         log_stable_prob=log_stable_prob)

    def make_move(self, move):
        new_tower, log_stable_prob = self.tower.play_move_log_probabilistic(move[0], move[1])
        return BasicState(tower=new_tower,
                          parent=self,
                          last_move=move,
                          log_stable_prob=log_stable_prob)

    def evaluate(self, fell=False, params=None):
        """
        this evaluation measures how good this state is to END in
            i.e. if player i plays a move that leads to this state,
                return 1 if player i will win and -1 if lost
            if this tower has no valid moves, the player that moves to this tower will win (as next player has no moves)
            if this tower falls, the player that moves to this tower will lose
        """  # this will be inverted up the tree, -1 reward for a loss, 1 reward for opp loss
        if params is None:
            params = dict()
        if fell:
            # this is if the tower fell
            return -1
        elif self.num_legal_moves == 0:
            # this is if the tower has no moves left
            return 1
        else:
            return random_playout(root_state=self,
                                  trials=params.get('trials', 1),
                                  depth_limit=params.get('depth_limit',float('inf')))


def random_playout(root_state: State, trials=1, depth_limit=float('inf')):
    """
    assumes root_state does not fall
    preform a random playout from the root state
        estimates how good root_state is to END at
    takes into account the probability of the root state falling
    Args:
        root_state: initial state
        trials: number of trials to take from root (default 1)
        depth_limit: depth to run to before returning 0
    Return:
        1 if ending at this state is always a win
        -1 if ending at this state is always a loss
        takes average if trials>1
    """
    if root_state.num_legal_moves == 0:
        return 1
    if depth_limit <= 0:
        return 0
    score = 0
    for trial in range(trials):
        # now we consider the next state
        next_move = root_state.moves[random.randint(0, root_state.num_legal_moves - 1)]
        next_state = root_state.make_move(next_move)
        if random.random() > math.exp(next_state.log_stable_prob):
            # next tower fell, so the player that ended at root_state won
            score += 1
            continue
        if next_state.num_legal_moves == 0:
            # no legal moves here, losing state for player that ended at root_state
            score += -1
            continue
        # otherwise we now have to evaluate the next state
        next_outcome = random_playout(next_state, trials=1, depth_limit=depth_limit - 1)
        score += -next_outcome
    return score/trials


class Node:
    def __init__(self, state, exploration_constant=math.sqrt(2), parent=None):
        self.state = state
        self.exploration_constant = exploration_constant
        self.parent = parent
        self.children = []
        self.last_child_idx = 0
        self.visits = 0
        self.cum_score = 0.0

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
            exploitation_term = child.get_exploit_score()
            exploration_term = math.sqrt(math.log(self.visits)/child.visits)
            score = exploitation_term + self.exploration_constant*exploration_term
            if score > best_score:
                best_child = child
                best_score = score

        return best_child

    def get_exploit_score(self):
        return self.cum_score/max(1, self.visits)

    def backpropagate(self, score):
        """
        Backpropagate the simulation result up the tree.

        Args:
            score: The score obtained from the simulation.
        """
        self.visits += 1
        self.cum_score += score

        if self.parent is not None:
            self.parent.backpropagate(-score)

    def tree_depth(self):
        if len(self.children) == 0:
            # we are a leaf node
            return 1
        return max(child.tree_depth() for child in self.children) + 1


def mcts_search(root_state,
                iterations,
                exploration_constant=2*math.sqrt(2),
                params=None,
                ):
    """
    Perform Monte Carlo Tree Search (MCTS) on the given root state to find the best action.
    assumes root_state is always non terminal
        i.e. never falls, and has legal actions

    Args:
        root_state: The initial state of the problem or game.
        iterations: The number of iterations to run the search.
        exploration_constant: The exploration constant (default: sqrt(2)).
        params: params to give evaluate

    Returns:
        The best action to take based on the MCTS algorithm.
    """
    # print('searching')
    root_node = Node(root_state, exploration_constant)
    for i in range(iterations):
        # if (i + 1)%1 == 0:
        #     print(f'\riteration {i + 1}', end='')
        node = root_node
        termination = None
        while True:
            if not node.is_fully_expanded():
                next_move = node.state.moves[node.last_child_idx]
                node.last_child_idx += 1
                node = node.add_child(node.state.make_move(next_move))
                termination = 'unexplored child node'
                break
            else:
                node = node.select_child()
                # print(f'{node.state.tower}\tlog stable prob = {node.state.log_stable_prob:.4f}')
            if random.random() > math.exp(node.state.log_stable_prob):
                termination = 'fell'
                break
            if node.state.num_legal_moves == 0:
                termination = 'no moves'
                break
        if termination == 'unexplored child node':
            # we have not done the fallen check yet in this case
            fell = random.random() > math.exp(node.state.log_stable_prob)
        else:
            fell = (termination == 'fell')

        simulation_result = node.state.evaluate(fell=fell, params=params)
        node.backpropagate(simulation_result)

    best_child = max(root_node.children, key=lambda x: x.get_exploit_score())
    return best_child.state.last_move, root_node


class MCTS_player(Agent):
    def __init__(self,
                 num_iterations=1000,
                 exploration_constant=2*math.sqrt(2),
                 ):
        """
        player for MCTS
        Args:
            num_iterations: number of iterations to run MCTS search
            exploration_constant: exploration constant to use during MCTS search
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.exploration_constant = exploration_constant

    def pick_move(self, tower: Tower):
        state = BasicState(tower)
        best_move, _ = mcts_search(root_state=state,
                                   iterations=self.num_iterations,
                                   exploration_constant=self.exploration_constant,
                                   )
        return best_move


# Example usage with a custom State class
if __name__ == "__main__":
    state = BasicState()
    player = 0
    is_terminal = False
    print(f"player {player}'s turn\t{state.tower} log_stable_prob={state.log_stable_prob:.4f}")
    num_moves = 0
    while random.random() <= math.exp(state.log_stable_prob):
        if state.num_legal_moves == 0:
            is_terminal = True
            break
        next_move, node = mcts_search(state, 1000, 2*math.sqrt(2))
        print('search tree depth:', node.tree_depth())
        state = state.make_move(next_move)
        player = 1 - player
        for child in node.children:
            exploitation_term = child.get_exploit_score()
            exploration_term = node.exploration_constant*math.sqrt(math.log(node.visits)/child.visits)
            score = exploitation_term + exploration_term
            print(
                f"{child.state.tower}\texploitation-term={exploitation_term:.4f} \texploration-term={exploration_term:.4f}\tscore={score:.4f}")
        print(f"player {player}'s turn\t{state.tower} log_stable_prob={state.log_stable_prob:.4f}")
        num_moves += 1
    if is_terminal:
        print(f'player {1 - player} won!')
    else:
        print(f'player {player} won!')
    print('game length', num_moves)
