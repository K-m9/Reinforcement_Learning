# Task04：DQN 算法及 Actor-Critic 算法
## 1. Q-learning
### 主要概念
Q-learning 算法本质上是在求解函数Q(s,a). 如下图，根据状态s和动作a, 得出在状态s下采取动作a会获得的未来的奖励，即Q(s,a)。 然后根据Q(s,a)的值，决定下一步动作该如何选择。
<div align=center><img height=300 src="https://upload-images.jianshu.io/upload_images/13326502-bdce1896ca4609a5.png"></div>

### 算法
Q-learning 算法中我们通过获得Q(s,a)函数来寻找在某个状态下的最好的动作，使得最终获得的累计奖励最大
其Q(s,a)的计算方法是利用贝尔曼方程
如下图是常见的两种形式：
<div align=center><img height=200 src="https://upload-images.jianshu.io/upload_images/13326502-32762146ff0afd71.png"></div>


参考文献：
https://www.jianshu.com/p/277abf64e369
