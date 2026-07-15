# 同一根杆子，两条路线——CartPole 里的 DQN 与 PPO

## 场景

一根杆子立在小车上。你可以往左或往右推车，保持杆子不倒。状态是四个连续值——位置、速度、角度、角速度。杆子倾斜超过 12° 或者车跑出边界，游戏结束。

这次不像 Cliff Walking——状态是连续的，没法给每个格子建一张 Q 表。你需要一个**会举一反三的神经网络**。

## 两条路线

同样是让杆子不倒，DQN 和 PPO 走了完全不同的两条路：

| | DQN | PPO |
|---|---|---|
| 学什么 | Q(s,a)：每个动作值多少钱 | π(a\|s)：每个动作该有多大概率 |
| 怎么选动作 | argmax Q（选最值钱的） | 按概率采样（训练时），取最大概率（评估时） |
| 数据怎么用 | off-policy，存 replay buffer 复用 | on-policy，用一次就扔 |
| 稳定靠什么 | Target Network + Replay Buffer | Clipped objective |

DQN 本质就是第一篇 Cliff Walking 里的 Q-Learning——只不过 Q 表换成了神经网络。PPO 则是第二篇 Frozen Lake 里 Policy Iteration 那个"评估→改进"框架的进化版——Critic 做评估，Actor 做改进。

## DQN：Q-Learning 的神经网络版，但需要"稳定器"

神经网络让事情变复杂了：一次梯度更新会影响很多状态的 Q 估计，不像表格 Q-Learning 每次只改一个格子。所以 DQN 加了两个稳定器：

```python
# 稳定器 1：Target Network —— 用延迟拷贝算 target，避免目标每步漂移
next_q_values = self.target_net(next_states).max(dim=1)[0]
targets = rewards + self.gamma * next_q_values * (1 - dones)

# 稳定器 2：Replay Buffer —— 随机抽旧经验，打破连续样本的相关性
states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
```

`diagnose.py` 的价值就在这——它逐个关掉稳定器让你亲眼看到：
- 关掉 Replay → 学不稳，方差飙升
- 关掉 Target Network → loss 爆炸
- epsilon 衰减太快 → 过早贪心，收敛到次优策略

## PPO：另一条完全不同的路

PPO 不学 Q 值，而是直接学"每个动作该有多大概率被选中"。核心约束只有一句话：**新策略不要离旧策略太远。**

```python
# PPO 的核心：ratio + clip
ratio = π_new(a|s) / π_old(a|s)           # 新旧策略的概率比
clipped = clamp(ratio, 0.8, 1.2)           # 限制在 [1-ε, 1+ε]
loss = -min(ratio * advantage, clipped * advantage)  # 取悲观估计
```

如果 advantage 是正的（这个动作比预期好），就提高它的概率——但不能一次提高太多。如果 advantage 是负的，就降低概率——同样不能降太猛。

PPO 不用 DQN 那种长期 replay buffer，因为它是 on-policy：旧数据太旧时，`π_old` 和 `π_new` 差太多，ratio 就失去意义了。

## 跑跑看

```bash
# DQN 消融实验——逐个拆掉稳定器看效果
python cartpole/diagnose.py

# PPO 训练
python cartpole/train_ppo.py

# 看训练好的 PPO 策略表演
python cartpole/train_ppo.py --demo
```

`diagnose.py` 会输出一张对比表——每个实验的训练前期/后期平均分、greedy 评估分、epsilon 终值。读这张表就是在读 DQN 的"体检报告"。

## 闭环

到这里，三条线闭合了：

```
Cliff Walking  → Q-Learning（表格 + 采样）
       ↓
Frozen Lake    → DP（表格 + 模型）
       ↓
CartPole       → DQN（Q-Learning + 神经网络 + 稳定器）
               → PPO（Policy Iteration + 神经网络 + clipping）
```

从表格到神经网络，从采样到计算再回到采样，从"学 Q 值"到"学策略"——RL 的主线走完了。
