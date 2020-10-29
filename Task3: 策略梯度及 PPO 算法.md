## Task3: 策略梯度及 PPO 算法
### 策略梯度算法
#### 基本思想
它让神经网络直接输出策略函数 π(s)，即在状态s下应该执行何种动作。对于非确定性策略，输出的是这种状态下执行各种动作的概率值，即如下的条件概率
<div align="center"><src="https://www.zhihu.com/equation?tex=%5Cpi%28a+%7C+s%29%3Dp%28a+%7C+s%29"></div>
