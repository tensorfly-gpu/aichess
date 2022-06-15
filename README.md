# aichess
# 使用alphazero算法打造属于你自己的象棋AI

## 一、每个文件的意义
collect.py      自我对弈用于数据收集

train.py    用于训练模型

game.py    实现象棋的逻辑

mcts.py    实现蒙特卡洛树搜索

paddle_net.py，pytorch_net.py   神经网络对走子进行评估

play_with_ai.py  人机对弈print版

UIplay.py   GUI界面人机对弈


## 二、提供了两个框架的版本进行训练：
使用pytorch版本请设置config.py 中CONFIG['use_frame'] = 'pytorch'，

使用pytorch版本请设置config.py 中CONFIG['use_frame'] = 'paddle'。

不管是使用哪个框架，都一定要安装gpu版本，而且要用英伟达显卡，因为我们蒙特卡洛一次走此要进行上千次的神经网络推理，所以必须要快！


## 三、本项目是多进程同步训练。
训练时，在终端运行python collect.py用于自我对弈产生数据，这个可以多开。

然后，在终端运行python train.py用于模型训练，这个终端只用开一个。

## 四、相关资源链接
B站视频链接：https://www.bilibili.com/video/BV183411g7GX

知乎文章：https://zhuanlan.zhihu.com/p/528824058?

aistudio上一键可运行项目：https://aistudio.baidu.com/aistudio/projectdetail/4215743 （可以使用免费的V100来进行训练）

## 五、参考与致谢
本项目主要参考的资料如下，十分感谢大佬们的分享
1、程世东 https://zhuanlan.zhihu.com/p/34433581 （中国象棋cchesszero ）

2、AI在打野 https://aistudio.baidu.com/aistudio/projectdetail/1403398 （用paddle打造的五子棋AI）

3、junxiaosong https://github.com/junxiaosong/AlphaZero_Gomoku (五子棋alphazero)

4、书籍：边做边学深度强化学习：PyTorch 程序设计实践

5、书籍：强化学习第二版

后续应该会对该AI继续训练下去，亲手造一个超强的下棋AI简直太酷了！
