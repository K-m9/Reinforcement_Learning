## Task3: 策略梯度及 PPO 算法
### 策略梯度算法
#### 基本思想
它让神经网络直接输出策略函数 π(s)，即在状态s下应该执行何种动作。对于非确定性策略，输出的是这种状态下执行各种动作的概率值，即如下的条件概率
<div align=center><img src="https://latex.codecogs.com/svg.latex?\pi(a|s)=p(a|s)"></div>
所谓确定性策略，是只在某种状态下要执行的动作是确定即唯一的，而非确定性动作在每种状态下要执行的动作是随机的，可以按照一定的概率值进行选择。这种做法的原理如下图所示。
<div align=center><img src="https://pic2.zhimg.com/80/v2-8ccec9899d4e2de9cc1e19654815e5bd_720w.jpg"></div>
此时的神经网络输出层的作用类似于多分类问题的softmax回归，输出的是一个概率分布，只不过这里的概率分布不是用来进行分类，而是执行动作。至于对于连续型的动作空间该怎么处理，我们在后面会解释。

因此，如果构造出了一个目标函数L，其输入是神经网络输出的策略函数 [公式] ，通过优化此目标函数，即可确定神经网络的参数θ，从而确定策略函数 [公式] 。这可以通过梯度上升法实现（与梯度下降法相反，向着梯度方向迭代，用于求函数的极大值）。训练时的迭代公式为
<div align=center><img src="https://www.zhihu.com/equation?tex=%5Cboldsymbol%7B%5Ctheta%7D_%7Bt%2B1%7D%3D%5Cboldsymbol%7B%5Ctheta%7D_%7Bt%7D%2B%5Calpha+%5Cnabla_%7B%5Cboldsymbol%7B%5Ctheta%7D%7D+L%5Cleft%28%5Cboldsymbol%7B%5Ctheta%7D_%7Bt%7D%5Cright%29"></div>

这里假设策略函数对参数的梯度![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Cmathfrak%7Bo%7D%7D+%5Cpi%28a+%7C+s+%3B+%5Cboldsymbol%7B%5Ctheta%7D%29)存在，从而保证![](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%5Cboldsymbol%7B%5Ctheta%7D%7D+L%5Cleft%28%5Cboldsymbol%7B%5Ctheta%7D_%7Bt%7D%5Cright%29) 。现在问题的核心就是如何构造这种目标函数L，以及如何生成训练样本。对于后者，采用了与DQN类似的思路，即按照当前策略随机地执行动作，并观察其回报值，以生成样本。

