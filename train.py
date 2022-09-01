"""使用收集到数据进行训练"""


import random
from collections import defaultdict, deque

import numpy as np
import pickle
import time

import zip_array
from config import CONFIG
from game import Game, Board
from mcts import MCTSPlayer
from mcts_pure import MCTS_Pure

if CONFIG['use_redis']:
    import my_redis, redis
    import zip_array

if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')


# 定义整个训练流程
class TrainPipeline:

    def __init__(self, init_model=None):
        # 训练参数
        self.board = Board()
        self.game = Game(self.board)
        self.n_playout = CONFIG['play_out']
        self.c_puct = CONFIG['c_puct']
        self.learn_rate = 1e-3
        self.lr_multiplier = 1  # 基于KL自适应的调整学习率
        self.temp = 1.0
        self.batch_size = CONFIG['batch_size']  # 训练的batch大小
        self.epochs = CONFIG['epochs']  # 每次更新的train_step数量
        self.kl_targ = CONFIG['kl_targ']  # kl散度控制
        self.check_freq = 100  # 保存模型的频率
        self.game_batch_num = CONFIG['game_batch_num']  # 训练更新的次数
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 500
        if CONFIG['use_redis']:
            self.redis_cli = my_redis.get_redis_cli()
        self.buffer_size = maxlen=CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)
        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                print('已加载上次最终模型')
            except:
                # 从零开始训练
                print('模型路径不存在，从零开始训练')
                self.policy_value_net = PolicyValueNet()
        else:
            print('从零开始训练')
            self.policy_value_net = PolicyValueNet()


    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2 + 1,
                                          is_shown=1)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio


    def policy_updata(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        # print(mini_batch[0][1],mini_batch[1][1])
        mini_batch = [zip_array.recovery_state_mcts_prob(data) for data in mini_batch]
        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype('float32')

        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch = np.array(mcts_probs_batch).astype('float32')

        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype('float32')

        # 旧的策略，旧的价值函数
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier
            )
            # 新的策略，新的价值函数
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1))
            if kl > self.kl_targ * 4:  # 如果KL散度很差，则提前终止
                break

        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        # print(old_v.flatten(),new_v.flatten())
        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.9f},"
               "explained_var_new:{:.9f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def run(self):
        """开始训练"""
        try:
            for i in range(self.game_batch_num):
                if not CONFIG['use_redis']:
                    while True:
                        try:
                            with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                                data_file = pickle.load(data_dict)
                                self.data_buffer = data_file['data_buffer']
                                self.iters = data_file['iters']
                                del data_file
                            print('已载入数据')
                            break
                        except:
                            time.sleep(30)
                else:
                    while True:
                        try:
                            l = len(self.data_buffer)
                            data = my_redis.get_list_range(self.redis_cli,'train_data_buffer', l if l == 0 else l - 1,-1)
                            self.data_buffer.extend(data)
                            self.iters = self.redis_cli.get('iters')
                            if self.redis_cli.llen('train_data_buffer') > self.buffer_size:
                                self.redis_cli.lpop('train_data_buffer',self.buffer_size/10)
                            break
                        except:
                            time.sleep(5)

                print('step i {}: '.format(self.iters))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_updata()
                    # 保存模型
                    if CONFIG['use_frame'] == 'paddle':
                        self.policy_value_net.save_model(CONFIG['paddle_model_path'])
                    elif CONFIG['use_frame'] == 'pytorch':
                        self.policy_value_net.save_model(CONFIG['pytorch_model_path'])
                    else:
                        print('不支持所选框架')

                time.sleep(CONFIG['train_update_interval'])  # 每10分钟更新一次模型

                if (i + 1) % self.check_freq == 0:
                    # win_ratio = self.policy_evaluate()
                    # print("current self-play batch: {},win_ratio: {}".format(i + 1, win_ratio))
                    # self.policy_value_net.save_model('./current_policy.model')
                    # if win_ratio > self.best_win_ratio:
                    #     print("New best policy!!!!!!!!")
                    #     self.best_win_ratio = win_ratio
                    #     # update the best_policy
                    #     self.policy_value_net.save_model('./best_policy.model')
                    #     if (self.best_win_ratio == 1.0 and
                    #             self.pure_mcts_playout_num < 5000):
                    #         self.pure_mcts_playout_num += 1000
                    #         self.best_win_ratio = 0.0
                    print("current self-play batch: {}".format(i + 1))
                    self.policy_value_net.save_model('models/current_policy_batch{}.model'.format(i + 1))
        except KeyboardInterrupt:
            print('\n\rquit')


if CONFIG['use_frame'] == 'paddle':
    training_pipeline = TrainPipeline(init_model='current_policy.model')
    training_pipeline.run()
elif CONFIG['use_frame'] == 'pytorch':
    training_pipeline = TrainPipeline(init_model='current_policy.pkl')
    training_pipeline.run()
else:
    print('暂不支持您选择的框架')
    print('训练结束')
