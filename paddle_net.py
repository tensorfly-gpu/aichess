"""策略价值网络"""


import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F


# 搭建残差块
class ResBlock(nn.Layer):

    def __init__(self, num_filters=256):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2D(num_features=num_filters)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2D(num_features=num_filters)
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return self.conv2_act(y)


# 搭建骨干网络，输入：N, 9, 10, 9 --> N, C, H, W
class Net(nn.Layer):

    def __init__(self, num_channels=256, num_res_blocks=13):
        super().__init__()
        # 初始化特征
        self.conv_block = nn.Conv2D(in_channels=9, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.conv_block_bn = nn.BatchNorm2D(num_features=256)
        self.conv_block_act = nn.ReLU()
        # 全局特征
        self.global_conv = nn.Conv2D(in_channels=9, out_channels=512, kernel_size=(10, 9))
        self.global_bn = nn.BatchNorm1D(512)
        # 残差块抽取特征
        self.res_blocks = nn.LayerList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
        # 策略头
        self.global_policy_fc = nn.Linear(512, 2086)
        self.policy_conv = nn.Conv2D(in_channels=num_channels, out_channels=16, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2D(16)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(16 * 9 * 10, 2086)
        # 价值头
        self.global_value_fc = nn.Linear(512, 256)
        self.value_conv = nn.Conv2D(in_channels=num_channels, out_channels=8, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2D(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8 * 9 * 10, 256)
        self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(256, 1)

    # 定义前向传播
    def forward(self, x):
        # 公共头
        global_x = self.global_conv(x)
        global_x = paddle.reshape(global_x, [-1, 512])
        global_x = self.global_bn(global_x)
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for layer in self.res_blocks:
            x = layer(x)
        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = paddle.reshape(policy, [-1, 16 * 10 * 9])
        policy = self.policy_fc(policy)
        global_policy = self.policy_act(self.global_policy_fc(global_x))
        policy = F.log_softmax(policy + global_policy)
        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = paddle.reshape(value, [-1, 8 * 10 * 9])
        global_value = self.value_act1(self.global_value_fc(global_x))
        value = self.value_fc1(value)
        value = self.value_act1(value)
        value = self.value_fc2(value + global_value)
        value = F.tanh(value)

        return policy, value


# 策略值网络，用来进行模型的训练
class PolicyValueNet:

    def __init__(self, model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.l2_const = 2e-3    # l2 正则化
        self.policy_value_net = Net()
        self.optimizer = paddle.optimizer.Adam(learning_rate=0.001,
                                               parameters=self.policy_value_net.parameters(),
                                               weight_decay=self.l2_const)
        if model_file:
            net_params = paddle.load(model_file)
            self.policy_value_net.set_state_dict(net_params)

    # 输入一个批次的状态，输出一个批次的动作概率和状态价值
    def policy_value(self, state_batch):
        self.policy_value_net.eval()
        state_batch = paddle.to_tensor(state_batch)
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.numpy())
        return act_probs, value.numpy()

    # 输入棋盘，返回每个合法动作的（动作，概率）元组列表，以及棋盘状态的分数
    def policy_value_fn(self, board):
        self.policy_value_net.eval()
        # 获取合法动作列表
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 9, 10, 9)).astype('float32')
        current_state = paddle.to_tensor(current_state)
        # 使用神经网络进行预测
        log_act_probs, value = self.policy_value_net(current_state)
        act_probs = np.exp(log_act_probs.numpy().flatten())
        # 只取出合法动作
        act_probs = zip(legal_positions, act_probs[legal_positions])
        # 返回动作概率，以及状态价值
        return act_probs, value.numpy()

    # 得到模型参数
    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    # 保存模型
    def save_model(self, model_file):
        net_params = self.get_policy_param()    # 取得模型参数
        paddle.save(net_params, model_file)

    # 执行一步训练
    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        self.policy_value_net.train()
        # 包装变量
        state_batch = paddle.to_tensor(state_batch)
        mcts_probs = paddle.to_tensor(mcts_probs)
        winner_batch = paddle.to_tensor(winner_batch)
        # 清零梯度
        self.optimizer.clear_gradients()
        # 设置学习率
        self.optimizer.set_lr(lr)
        # 前向运算
        log_act_probs, value = self.policy_value_net(state_batch)
        value = paddle.reshape(x=value, shape=[-1])
        # 价值损失
        value_loss = F.mse_loss(input=value, label=winner_batch)
        # 策略损失
        policy_loss = -paddle.mean(paddle.sum(mcts_probs * log_act_probs, axis=1))  # 希望两个向量方向越一致越好
        # 总的损失，注意l2惩罚已经包含在优化器内部
        loss = value_loss + policy_loss
        # 反向传播及优化
        loss.backward()
        self.optimizer.minimize(loss)
        # 计算策略的熵，仅用于评估模型
        entropy = -paddle.mean(
            paddle.sum(paddle.exp(log_act_probs) * log_act_probs, axis=1)
        )
        return loss.numpy(), entropy.numpy()[0]


if __name__ == '__main__':
    net = Net()
    test_data = paddle.ones([8, 9, 10, 9])
    x_act, x_val = net(test_data)
    print(x_act.shape)  # 8, 2086
    print(x_val.shape)  # 8, 1
