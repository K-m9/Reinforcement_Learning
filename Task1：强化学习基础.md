# Task01：强化学习基础

## 学习视频与笔记
- 概括与基础 上：https://www.bilibili.com/video/BV1LE411G7Xj
- 概括与基础 下：https://www.bilibili.com/video/BV1g7411Z7SJ
- 李宏毅深度强化学习笔记(LeeDeepRL-Notes)：
https://datawhalechina.github.io/leedeeprl-notes/#

## 其他学习资料
- gym官网：http://gym.openai.com/envs/#classic%20control
- zhoubolei/introRL：https://github.com/zhoubolei/introRL
- cuhkrlcourse/RLexample：https://github.com/cuhkrlcourse/RLexample

## 习题
- 1. 强化学习得基本结构是什么？

强化学习是智能体agent与环境environment交互的一个学习过程。
具体而言，agent通过接收environment发出的状态state进行决策action，
action传入environment进行试验，得到奖励reward和新的state，
为了最大化reward，agent需要根据state不断地调整action再传入environment
进行试错。

- 2. 强化学习相对于监督学习为什么训练会更加困难？（强化学习的特征）

1. 监督学习的样本数据是独立的，而强化学习的输入的序列数据不是独立的。
2. 强化学习过程中，每一步的正确行为是未知的，需要通过不断地试错，而这会消耗大量成本。
3. 强化学习的奖励信号（reward signal）是延迟的，需要等每轮行为结束后才给出。

- 3. 强化学习的基本特征有哪些？

1. 试错 trial-and-error exploration，需要不断探索来熟悉环境并找到规律。
2. 奖励延迟 delayed reward，需每轮行为结束后才能知道奖励。
3. 时间问题 time matters，数据非i.i.d分布，而且是与时间有关联，因此时间很重要。
4. agent的行为会影响它随后得到的数据。

- 4. 近几年强化学习发展迅速的原因？

1. 算力（GPU、TPU）的提升，我们可以更快地做更多的 trial-and-error 的尝试来使得Agent在Environment里面获得很多信息，取得更大的Reward。
2. 我们有了深度强化学习这样一个端到端的训练方法，可以把特征提取和价值估计或者决策一起优化，这样就可以得到一个更强的决策网络。

- 5. 状态和观测有什么关系？

状态是一个环境或情况的所有方面进行完整描述，而观测是对状态进行部分描述，仅限于可以观测得到的方面。状态包含观测。

- 6. 对于一个强化学习 Agent，它由什么组成？

1. policy function：决策函数，agent会根据该函数进行下一步决策action，包括随机性策略（stochastic policy）和确定性策略（deterministic policy）；
2. value functio：价值函数，对当前状态进行估价reward；
3. model：模型，表示agent对当前环境的理解情况，它决定了这个世界是如何进行的。

- 7. 根据强化学习 Agent 的不同，我们可以将其分为哪几类？

1. value-based agent：基于价值函数的agent，这一类agent显示地学习价值函数，隐式学习它的策略。因为这个策略是从我们学到的价值函数里面推算出来的。
2. policy-based agent：基于策略的Agent，它直接去学习 policy，就是说你直接给它一个 state，它就会输出这个动作的概率。然后在这个 policy-based agent 里面并没有去学习它的价值函数。
3. actor-critic agent：两者结合，把 value-based 和 policy-based 结合起来就有了 Actor-Critic agent。这一类 Agent 就把它的策略函数和价值函数都学习了，然后通过两者的交互得到一个更佳的状态。

- 8. 基于策略迭代和基于价值迭代的强化学习方法有什么区别?

1. 基于策略迭代的强化学习方法，agent会制定一套动作策略（确定在给定状态下需要采取何种动作），并根据这个策略进行操作。强化学习算法直接对策略进行优化，使制定的策略能够获得最大的奖励；基于价值迭代的强化学习方法，agent不需要制定显式的策略，它维护一个价值表格或价值函数，并通过这个价值表格或价值函数来选取价值最大的动作。
**简单的说，policy-based agent重点想要学会一个动作，而value-based agent重点想要获得最大的奖励。**
2. 基于价值迭代的方法只能应用在不连续的、离散的环境下（如围棋或某些游戏领域），对于行为集合规模庞大、动作连续的场景（如机器人控制领域），其很难学习到较好的结果（此时基于策略迭代的方法能够根据设定的策略来选择连续的动作)；
3. 基于价值迭代的强化学习算法有 Q-learning、 Sarsa 等，而基于策略迭代的强化学习算法有策略梯度算法等。
4. 此外， Actor-Critic 算法同时使用策略和价值评估来做出决策，其中，智能体会根据策略做出动作，而价值函数会对做出的动作给出价值，这样可以在原有的策略梯度算法的基础上加速学习过程，取得更好的效果。

- 9. 有模型（model-based）学习和免模型（model-free）学习有什么区别？

针对是否需要对真实环境建模，强化学习可以分为有模型学习和免模型学习。
有模型学习是指根据环境中的经验，构建一个虚拟世界，同时在真实环境和虚拟世界中学习；
免模型学习是指不对环境进行建模，直接与真实环境进行交互来学习到最优策略。
总的来说，有模型学习相比于免模型学习仅仅多出一个步骤，即对真实环境进行建模。
免模型学习通常属于数据驱动型方法，需要大量的采样来估计状态、动作及奖励函数，从而优化动作策略。
免模型学习的泛化性要优于有模型学习，原因是有模型学习算需要对真实环境进行建模，并且虚拟世界与真实环境之间可能还有差异，这限制了有模型学习算法的泛化性。

- 10. 强化学习的通俗理解

environment 跟 reward function 不是我们可以控制的，environment 跟 reward function 是在开始学习之前，就已经事先给定的。
我们唯一能做的事情是调整 actor 里面的 policy，使得 actor 可以得到最大的 reward。
Actor 里面会有一个 policy， 这个 policy 决定了actor 的行为。
Policy 就是给一个外界的输入，然后它会输出 actor 现在应该要执行的行为。
