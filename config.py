CONFIG = {
    'kill_action': 60,
    'dirichlet': 0.15,       # 国际象棋，0.3；日本将棋，0.15；围棋，0.03
    'play_out': 20,        # 每次移动的模拟次数
    'c_puct': 5,             # u的权重
    'buffer_size': 100000,   # 经验池大小
    'paddle_model_path': 'current_policy.model',      # paddle模型路径
    'pytorch_model_path': 'current_policy.pkl',   # pytorch模型路径
    'train_data_buffer_path': 'train_data_buffer.pkl',   # 数据容器的路径
    'batch_size': 512,  # 每次更新的train_step数量
    'kl_targ': 0.02,  # kl散度控制
    'epochs' : 5,  # 每次更新的train_step数量
    'game_batch_num': 3000,  # 训练更新的次数
    'use_frame': 'pytorch',  # paddle or pytorch根据自己的环境进行切换
}