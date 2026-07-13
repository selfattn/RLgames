# CartPole: DQN and PPO

CartPole 是本仓库的深度 RL 环境。它的状态是连续的：

```text
[x, x_dot, theta, theta_dot]
```

所以不能像 Cliff Walking 那样为每个状态动作对建一张 Q-table。

## DQN

DQN 是 value-based 方法：

```text
输入 state
输出 Q(left), Q(right)
```

动作来自：

```text
argmax_a Q(s,a)
```

训练时使用 Q-Learning 风格的 TD target：

```text
r + gamma max_a Q_target(s',a)
```

DQN 的关键稳定器是：

```text
Replay Buffer
Target Network
```

运行：

```bash
python3 rl_playground/cartpole/train_play.py
python3 rl_playground/cartpole/diagnose.py
```

## PPO

PPO 是 policy-based / actor-critic 方法。

先把两个前置概念放在文本里，不单独实现：

```text
REINFORCE:
  直接提高高回报轨迹里那些动作的概率。
  问题是方差很高。

Actor-Critic:
  actor 输出动作概率 pi(a|s)。
  critic 估计 V(s)，帮助 actor 判断动作比预期好还是差。
```

PPO 在 Actor-Critic 之上加了一个稳定性约束：

```text
新策略不要离旧策略太远。
```

它通过 clipped objective 实现：

```text
ratio = pi_new(a|s) / pi_old(a|s)
clip ratio to [1 - epsilon, 1 + epsilon]
```

所以 PPO 的核心不是“更新策略”，而是：

```text
小步、稳定地更新策略。
```

在这个实现里：

```text
actor: 输出 P(left), P(right)
critic: 输出 V(state)
```

运行：

```bash
python3 rl_playground/cartpole/train_ppo.py
```

如果想看训练后的 ASCII 演示：

```bash
python3 rl_playground/cartpole/train_ppo.py --demo
```

快速 smoke run：

```bash
python3 rl_playground/cartpole/train_ppo.py --updates 3 --steps-per-update 128
```

## DQN vs PPO

| 方法 | 学什么 | 怎么选动作 | 数据使用方式 | 稳定性关键 |
|---|---|---|---|---|
| DQN | `Q(s,a)` | `argmax Q(s,a)` + epsilon 探索 | off-policy，可 replay 旧经验 | replay buffer + target network |
| PPO | `pi(a|s)` 和 `V(s)` | 从 actor 概率采样，评估时取最大概率动作 | on-policy，使用新采样 rollout | clipped objective + advantage normalization |

一句话：

```text
DQN 学“哪个动作值更高”；
PPO 学“每个动作应该有多大概率被选中”。
```
