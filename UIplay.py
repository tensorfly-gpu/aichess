import pygame
import sys
import copy
import random
from game import move_action2move_id, Game, Board
from mcts import MCTSPlayer
import time
from config import CONFIG



if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')


class Human:

    def __init__(self):
        self.agent = 'HUMAN'

    def get_action(self, move):
        # move从鼠标点击事件触发
        # print('当前是player2在操作')
        # print(board.current_player_color)
        if  move_action2move_id.__contains__(move):
            move = move_action2move_id[move]
        else:
            move = -1
        # move = random.choice(board.availables)
        return move

    def set_player_ind(self, p):
        self.player = p


if CONFIG['use_frame'] == 'paddle':
    policy_value_net = PolicyValueNet(model_file='current_policy.model')
elif CONFIG['use_frame'] == 'pytorch':
    policy_value_net = PolicyValueNet(model_file='current_policy.pkl')
else:
    print('暂不支持您选择的框架')

# 初始化pygame
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load('bgm/yinzi.ogg')
pygame.mixer.music.set_volume(0.03)
pygame.mixer.music.play(loops=-1, start=0.0)

size = width, height = 700, 700
bg_image = pygame.image.load('imgs/board.png')  #图片位置
bg_image = pygame.transform.smoothscale(bg_image, size)

clock = pygame.time.Clock()
fullscreen = False
# 创建指定大小的窗口
screen = pygame.display.set_mode(size)
# 设置窗口标题
pygame.display.set_caption("中国象棋")

# 加载一个列表进行图像的绘制
# 列表表示的棋盘初始化，红子在上，黑子在下，禁止对该列表进行编辑，使用时必须使用深拷贝
board_list_init = [['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '红炮', '一一', '一一', '一一', '一一', '一一', '红炮', '一一'],
                   ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
                   ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']]

# 加载棋子被选中的图片
fire_image = pygame.transform.smoothscale(pygame.image.load("imgs/fire.png").convert_alpha(), (width // 10, height // 10))
fire_image.set_alpha(200)

# 制作一个从字符串到pygame.surface对象的映射
str2image = {
    '红车': pygame.transform.smoothscale(pygame.image.load("imgs/hongche.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红马': pygame.transform.smoothscale(pygame.image.load("imgs/hongma.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红象': pygame.transform.smoothscale(pygame.image.load("imgs/hongxiang.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红士': pygame.transform.smoothscale(pygame.image.load("imgs/hongshi.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红帅': pygame.transform.smoothscale(pygame.image.load("imgs/hongshuai.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红炮': pygame.transform.smoothscale(pygame.image.load("imgs/hongpao.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '红兵': pygame.transform.smoothscale(pygame.image.load("imgs/hongbing.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑车': pygame.transform.smoothscale(pygame.image.load("imgs/heiche.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑马': pygame.transform.smoothscale(pygame.image.load("imgs/heima.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑象': pygame.transform.smoothscale(pygame.image.load("imgs/heixiang.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑士': pygame.transform.smoothscale(pygame.image.load("imgs/heishi.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑帅': pygame.transform.smoothscale(pygame.image.load("imgs/heishuai.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑炮': pygame.transform.smoothscale(pygame.image.load("imgs/heipao.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
    '黑兵': pygame.transform.smoothscale(pygame.image.load("imgs/heibing.png").convert_alpha(), (width // 10 - 10, height // 10 - 10)),
}
str2image_rect = {
    '红车': str2image['红车'].get_rect(),
    '红马': str2image['红马'].get_rect(),
    '红象': str2image['红象'].get_rect(),
    '红士': str2image['红士'].get_rect(),
    '红帅': str2image['红帅'].get_rect(),
    '红炮': str2image['红炮'].get_rect(),
    '红兵': str2image['红兵'].get_rect(),
    '黑车': str2image['黑车'].get_rect(),
    '黑马': str2image['黑马'].get_rect(),
    '黑象': str2image['黑象'].get_rect(),
    '黑士': str2image['黑士'].get_rect(),
    '黑帅': str2image['黑帅'].get_rect(),
    '黑炮': str2image['黑炮'].get_rect(),
    '黑兵': str2image['黑兵'].get_rect(),
}


# 根据棋盘列表获得最新位置
# 返回一个由image和rect对象组成的列表
x_ratio = 80
y_ratio = 72
x_bais = 30
y_bais = 25
def board2image(board):
    return_image_rect = []
    for i in range(10):
        for j in range(9):
            if board[i][j] == '红车':
                str2image_rect['红车'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['红车'])
                return_image_rect.append((str2image['红车'], str2image_rect_copy))
            elif board[i][j] == '红马':
                str2image_rect['红马'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['红马'])
                return_image_rect.append((str2image['红马'], str2image_rect_copy))
            elif board[i][j] == '红兵':
                str2image_rect['红兵'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['红兵'])
                return_image_rect.append((str2image['红兵'], str2image_rect_copy))
            elif board[i][j] == '红象':
                str2image_rect['红象'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['红象'])
                return_image_rect.append((str2image['红象'], str2image_rect_copy))
            elif board[i][j] == '红炮':
                str2image_rect['红炮'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['红炮'])
                return_image_rect.append((str2image['红炮'], str2image_rect_copy))
            elif board[i][j] == '红士':
                str2image_rect['红士'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['红士'])
                return_image_rect.append((str2image['红士'], str2image_rect_copy))
            elif board[i][j] == '红帅':
                str2image_rect['红帅'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['红帅'])
                return_image_rect.append((str2image['红帅'], str2image_rect_copy))
            elif board[i][j] == '红兵':
                str2image_rect['红兵'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['红兵'])
                return_image_rect.append((str2image['红兵'], str2image_rect_copy))
            elif board[i][j] == '黑车':
                str2image_rect['黑车'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['黑车'])
                return_image_rect.append((str2image['黑车'], str2image_rect_copy))
            elif board[i][j] == '黑马':
                str2image_rect['黑马'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['黑马'])
                return_image_rect.append((str2image['黑马'], str2image_rect_copy))
            elif board[i][j] == '黑兵':
                str2image_rect['黑兵'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['黑兵'])
                return_image_rect.append((str2image['黑兵'], str2image_rect_copy))
            elif board[i][j] == '黑象':
                str2image_rect['黑象'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['黑象'])
                return_image_rect.append((str2image['黑象'], str2image_rect_copy))
            elif board[i][j] == '黑炮':
                str2image_rect['黑炮'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['黑炮'])
                return_image_rect.append((str2image['黑炮'], str2image_rect_copy))
            elif board[i][j] == '黑士':
                str2image_rect['黑士'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['黑士'])
                return_image_rect.append((str2image['黑士'], str2image_rect_copy))
            elif board[i][j] == '黑帅':
                str2image_rect['黑帅'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['黑帅'])
                return_image_rect.append((str2image['黑帅'], str2image_rect_copy))
            elif board[i][j] == '黑兵':
                str2image_rect['黑兵'].center = (j * x_ratio + x_bais, i * y_ratio + y_bais)
                str2image_rect_copy = copy.deepcopy(str2image_rect['黑兵'])
                return_image_rect.append((str2image['黑兵'], str2image_rect_copy))
    return return_image_rect


fire_rect = fire_image.get_rect()
fire_rect.center = (0 * x_ratio + x_bais, 3 * y_ratio + y_bais)


# 加载两个玩家，AI对AI，或者AI对human
board=Board()
# 开始的玩家
start_player = 1

player1 = MCTSPlayer(policy_value_net.policy_value_fn,
                     c_puct=5,
                     n_playout=1000,
                     is_selfplay=0)
player2 = MCTSPlayer(policy_value_net.policy_value_fn,
                     c_puct=5,
                     n_playout=2000,
                     is_selfplay=0)


# player2 = Human()

board.init_board(start_player)
p1, p2 = 1, 2
player1.set_player_ind(1)
player2.set_player_ind(2)
players = {p1: player1, p2: player2}


# 切换玩家
swicth_player = True
draw_fire = False
move_action = ''
first_button = False
while True:

    # 填充背景
    screen.blit(bg_image, (0, 0))
    for image, image_rect in board2image(board=board.state_deque[-1]):
        screen.blit(image, image_rect)
    if draw_fire:
        screen.blit(fire_image, fire_rect)
    # 更新界面
    pygame.display.update()
    # 不高于60帧
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:    #按下按键
            # print("[MOUSEBUTTONDOWN]", event.pos, event.button)
            mouse_x, mouse_y = event.pos
            if not first_button:
                for i in range(10):
                    for j in range(9):
                        if abs(80 * j + 30 - mouse_x) < 30 and abs(72 * i + 25 - mouse_y) < 30:
                            first_button = True
                            start_i_j = j, i
                            fire_rect.center = (start_i_j[0] * x_ratio + x_bais, start_i_j[1] * y_ratio + y_bais)
                            # screen.blit(fire_image, fire_rect)
                            break

            elif first_button:
                for i in range(10):
                    for j in range(9):
                        if abs(80 * j + 30 - mouse_x) < 30 and abs(72 * i + 25 - mouse_y) < 30:
                            first_button = False
                            end_i_j = j, i
                            move_action = str(start_i_j[1]) + str(start_i_j[0]) + str(end_i_j[1]) + str(end_i_j[0])
                            # screen.blit(fire_image, fire_rect)

    if swicth_player:
        current_player = board.get_current_player_id()  # 红子对应的玩家id
        player_in_turn = players[current_player]  # 决定当前玩家的代理

    if player_in_turn.agent == 'AI':
        pygame.display.update()
        start_time = time.time()
        move = player_in_turn.get_action(board)  # 当前玩家代理拿到动作
        print('耗时：', time.time() - start_time)
        board.do_move(move)  # 棋盘做出改变
        swicth_player = True
        draw_fire = False
    elif player_in_turn.agent == 'HUMAN':
        draw_fire = True
        swicth_player = False
        if len(move_action) == 4:
            move = player_in_turn.get_action(move_action)  # 当前玩家代理拿到动作
            if move != -1:
                board.do_move(move)  # 棋盘做出改变
                swicth_player = True
                move_action = ''
                draw_fire = False

    end, winner = board.game_end()
    if end:
        if winner != -1:
            print("Game end. Winner is", players[winner])
        else:
            print("Game end. Tie")
        sys.exit()