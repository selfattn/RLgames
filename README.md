# 三个实验：悬崖、冰面与倒立摆，不用背公式的 RL 入门

我研究生第一次上 RL 课，是从 Bellman 方程和 UCB 开始看的。

老师讲得很好，公式我也都抄了。但问题是——我不知道这些公式各自在解决什么问题，更不知道它们之间是什么关系。Value Iteration 和 Q-Learning 都叫"值函数"，但一个要先知道环境模型，一个是采样——当时没人告诉我这根线是怎么连起来的。

期末考试一过，脑子里只剩一片模糊。

后来我发现，RL 的核心问题其实只有三个：

1. **如果我不知道环境规则，只能靠试错，怎么学？**
2. **如果我知道环境规则呢？能在家算好再出门吗？**
3. **如果状态多到表都装不下呢？**

这三个问题，对应三个经典实验：

| 实验 | 环境 | 核心问题 | 你会看到什么 |
|---|---|---|---|
| **悬崖** | Cliff Walking | 无模型，靠采样 | 贴悬崖冒险 vs 绕远路保守——同一行代码的两种人生 |
| **冰面** | Frozen Lake | 有模型，靠计算 | 一口气算到底 vs 慢慢评估改进——两条路到达同一个终点 |
| **倒立摆** | CartPole | 表格装不下，上网络 | Q-Learning + 神经网络 = 能跑，但两个"稳定器"缺一个就崩 |

这不是从公式出发的教程。代码已经写好了，你要做的是跑起来、看现象、改参数。

## 快速开始

```bash
git clone git@github.com:selfattn/RLgames.git
cd RLgames

pip install numpy torch
```

然后运行三个诊断脚本：

```bash
# 悬崖：看 Q-Learning 和 SARSA 的路径差异
python cliff_walking/diagnose.py

# 冰面：看 Value Iteration 和 Policy Iteration 的收敛过程
python frozen_lake/diagnose.py

# 倒立摆：看 DQN 稳定器逐个拆掉会怎样
python cartpole/diagnose.py
```

每个脚本跑完都会输出对比报告——不需要你写一行 RL 代码，只需要读输出、看现象。

## 还有更多玩法

```bash
# 动态演示 Value Iteration 在冰面上传播
python frozen_lake/train_play.py --mode value

# 动态演示 Policy Iteration 的评估-改进循环
python frozen_lake/train_play.py --mode policy

# PPO 训练倒立摆
python cartpole/train_ppo.py

# PPO 训练完看它表演
python cartpole/train_ppo.py --demo
```

## 读完这些，你不会背 Bellman 方程

但你会知道——Q-Learning 那一行 `max` 到底在干什么，为什么去掉它就成了 SARSA，Policy Iteration 那个"评估→改进"的循环怎么一路演化成了 PPO 的 Actor-Critic。

RL 的地图，这次从直觉出发。

## 文章系列

- [悬崖边上的两种走法: Q-Learning 与 SARSA](./posts/01-cliff-walking.md)
