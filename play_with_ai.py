import random
from game import move_action2move_id, Game, Board
from net import PolicyValueNet
from mcts import MCTSPlayer


# 测试Board中的start_play
class Human1:
    def get_action(self, board):
        move = move_action2move_id[input('请输入')]
        # move = random.choice(board.availables)
        return move

    def set_player_ind(self, p):
        self.player = p


policy_value_net = PolicyValueNet(model_file='current_policy100.model')

mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                c_puct=5,
                                n_playout=100,
                                is_selfplay=0)

human = Human1()


game = Game(board=Board())
game.start_play(mcts_player, human, start_player=1, is_shown=1)

