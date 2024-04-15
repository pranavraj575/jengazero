from mcts.base.base import BaseState
from mcts.searcher.mcts import MCTS


class MyState(BaseState):
    
    def get_possible_actions(self) -> [any]: # type: ignore
        pass

    def take_action(self, action: any) -> 'BaseState':
        pass

    def is_terminal(self) -> bool:
        pass

    def get_reward(self) -> float:
        pass

    def get_current_player(self) -> int:
        pass


initial_state = MyState()

searcher = MCTS(time_limit=1000)
bestAction = searcher.search(initial_state=initial_state)