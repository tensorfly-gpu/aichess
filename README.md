# aichess
# 使用alphazero算法打造属于你自己的象棋AI


提供了两个框架的版本进行训练：

使用pytorch版本请设置config.py 中CONFIG['use_frame'] = 'pytorch'，

使用pytorch版本请设置config.py 中CONFIG['use_frame'] = 'paddle'。


本项目是多进程同步训练。

训练时，在终端运行python collect.py用于自我对弈产生数据，这个可以多开。

然后，在终端运行python train.py用于模型训练，这个终端只用开一个。


B站视频链接：可以参考

aistudio上一键可运行项目：https://aistudio.baidu.com/aistudio/projectdetail/4215743 （可以使用免费的V100来进行训练）
