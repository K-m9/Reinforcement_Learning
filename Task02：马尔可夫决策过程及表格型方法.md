# Task02：马尔可夫决策过程及表格型方法

## 学习目标
1. 通过理解马尔可夫链和马尔可夫奖励过程(MDP)来理解马尔可夫决策过程(MDP)
2. 介绍MDP中的policy evaluation，即当给定一个决策过后，如何计算价值函数
3. 介绍MDP中的控制，具体有两种算法：policy iteration & value iteration

## 学习视频和笔记
- 马尔科夫决策过程 上：https://www.bilibili.com/video/BV1g7411m7Ms
- 马尔科夫决策过程 下：https://www.bilibili.com/video/BV1u7411m7rh
- MDP、Q 表格：https://www.bilibili.com/video/BV1yv411i7xd?p=4
- 强化概念、TD更新、Sarsa引入：https://www.bilibili.com/video/BV1yv411i7xd?p=5
- Sarsa算法介绍与代码解析：https://www.bilibili.com/video/BV1yv411i7xd?p=6
- on_policy与off_policy对比、Q-learning解析：https://www.bilibili.com/video/BV1yv411i7xd?p=7
- 学习笔记：https://datawhalechina.github.io/leedeeprl-notes/#/chapter2/chapter2

## 学习内容

<div align=center><img width="590" height="200" src="https://pic4.zhimg.com/v2-991ab4b45d8f1d6340adc9313c13e91b_1200x500.jpg"/></div>
<p align="center">图1 强化学习过程</p>
强化学习中agent与环境的交互过程可以由马尔可夫决策过程表示，因此MDP是强化学习的基本框架。<br>
在MDP中，环境是全部可观测的（fully observable），但在许多情况下，环境中的许多状态是不可观测的，
而部分观测的问题也可以转化成MDP问题。

### 1. 马尔可夫模型
#### 1.1 马尔可夫过程(MP)
马尔可夫性（无后效性）：过程或（系统）在时刻t0所处的状态为已知的条件下，过程在时刻t > t0所处状态的条件分布，与过程在时刻t0之前处的状态无关。<br>

**马尔可夫性质是所有马尔可夫过程的基础。** 具有马尔可夫性的随机过程称为马尔可夫过程。用分布函数表述马尔可夫过程：<br>
设I：随机过程![](https://latex.codecogs.com/svg.latex?X(t),t\in%20T)的状态空间，如果对时间t的任意n个数值:
<div align=center><img width="800" height="20" src="https://latex.codecogs.com/svg.latex?\begin{align*}P\{X(t_n)\leq%20x_n|X(t_1)=x_1,X(t_2)=x_2,...,X(t_{n-1})=x_{n-1}\}=%20P\{X(t_n)\leq%20x_n|X(t_{n-1})=x_{n-1}\}\end{align*}"/></div>
或<div align=center><img width="800" height="20" src="https://latex.codecogs.com/svg.latex?p(x_{t+1}|x_t)=p(x_{t+1}|x_t,x_{t-1},...,x_1)"/></div>

状态转移矩阵：
<div align=center><img width="360" height="150" src="https://tse1-mm.cn.bing.net/th/id/OIP.2FuKaTIVpZRHhWsYNHu0twHaDO?pid=Api&rs=1"/></div>

其中![](https://latex.codecogs.com/svg.latex?p^{(n)}_{ij}=p^{(n)}(x_i|x_j))表示第n步从状态i转移到状态j的概率

#### 1.2 马尔可夫奖励过程(MRPs)
##### 1.2.1 定义
MRP = 马尔可夫链 + 一个奖励函数
- S：状态集合（![](https://latex.codecogs.com/svg.latex?s%20\in%20S)）
- P：状态转移矩阵 ![](https://latex.codecogs.com/svg.latex?p(s_{t+1}=s%27|s_t=s))
- R：奖励函数![](https://latex.codecogs.com/svg.latex?R(s_t=s)=E[r_t|s_t=s])，指到达某状态可获得的奖励
- ![](https://latex.codecogs.com/svg.latex?\gamma)：折扣系数![](https://latex.codecogs.com/svg.latex?\gamma\in[0,1])

#### 1.3 马尔可夫决策过程(MDPs)


## 习题
1. 为什么在马尔可夫奖励过程中需要discounted factor?
- 1. 由于有些马尔可夫过程是带环的，因此需要限定奖励在有限范围内。
- 2. 由于奖励是否获得存在不确定性，希望尽快得到奖励。
- 3. 由于我们对奖励的渴望程度符合边际效用递减规律，因此希望尽快得到奖励。

2. 为什么矩阵形式的Bellman Equation的解析解比较难解？<br>
Bellman Equation的解析解公式为:![](https://latex.codecogs.com/svg.latex?V=(1-\gamma%20P)^{-1}R)
但这个矩阵求逆的复杂度为![](https://latex.codecogs.com/svg.latex?O(N^3)),其中N是状态个数。
若状态个数很多的情况下，矩阵求逆非常困难。

3. 计算贝尔曼等式（Bellman Equation）的常见方法以及区别？
- 1. Monte Carlo Algorithm（蒙特卡罗方法），随机生成大量轨迹，求各轨迹价值函数均值。
- 2. Iterative Algorithm（动态规划方法），不断更新价值函数直至当前价值函数与上一价值函数差值小于给定阈值。
- 3. Temporal-Difference Learning（以上两者的结合方法）

4. 马尔可夫奖励过程（MRP）与马尔可夫决策过程 （MDP）的区别？
- 1. MDP比MRP多了一个决策action参数，因此MDP的状态转移矩阵也多了一个condition。
- 2. 在MDP中，若给定一个policy function $\pi$, MDP可转化为MRP。

5. MDP 里面的状态转移跟 MRP 以及 MP 的结构或者计算方面的差异？<br>
对于MRP和MP，状态转移过程只取决于当前状态state；
而MDP多了一层action决策，即在当前状态下，先判定作出什么决策action，根据action再得出未来状态的概率分布。

6. 我们如何寻找最佳的policy，方法有哪些？
- 1. policy search（穷举法）：假定每个状态有A种行为策略，则有![](https://latex.codecogs.com/svg.latex?|A|^{|S|})种policy，效率过低。
- 2. policy iteration：<div align="center"><img width="600" src="https://programmingbeenet.files.wordpress.com/2019/03/policy_iteration.png">
- 3. Value iteration: <div align="center"><img width="600" src="https://danieltakeshi.github.io/assets/cs287_value_iteration.png">



