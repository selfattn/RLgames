# RL Playground Insights

这份文件是速查提纲：帮你抓住每个算法在解决什么问题、最容易错在哪里。

## 1. 总路线

| 阶段 | 环境 | 算法 | 重点 |
|---|---|---|---|
| DP | Frozen Lake | Value Iteration / Policy Iteration | 已知模型，直接规划 |
| TD | Cliff Walking | SARSA / Q-Learning | 无模型，采样学习 |
| Deep Value | CartPole | DQN | 用网络近似 Q 值 |
| Policy Gradient | CartPole | PPO | 直接学习动作概率 |

一句话：

```text
DP: 有模型，算期望。
TD: 没模型，用采样。
DQN: Q-Learning + 神经网络 + 稳定器。
PPO: Actor-Critic + clipping。
```

## 2. 为什么这些 demo 仍然有价值

AI 会写 RL 代码，但 RL 常见问题是：

```text
代码能跑，曲线也在动，但学到的是错的。
```

要重点警惕：

- reward / done 写错。
- SARSA 和 Q-Learning target 写反。
- DQN 没有 replay buffer / target network。
- epsilon decay 太快。
- 只看训练最高分，不看 greedy eval。
- PPO 策略更新太猛，训练不稳定。

这些 demo 的目标不是背代码，而是训练判断力。

## 3. DP vs TD

DP 需要完整环境模型：

```text
P(s', r | s, a)
```

TD 只需要真实经验：

```text
s, a, r, s'
```

对比：

```text
DP = full backup，枚举所有可能结果。
TD = sample backup，只用采样到的一步。
```

## 4. Frozen Lake: VI vs PI

`get_transitions(state, action)` 是 DP 的入口。它枚举：

```text
在状态 s 做动作 a 后，所有可能的 s'、reward、done 和概率。
```

注意转移依赖动作：

```text
P(s' | s, a)，不是 P(s' | s)。
```

### Value Iteration

直接逼近最优价值：

```text
V(s) <- max_a E[r + gamma V(s')]
```

直觉：每次都假设下一步选最优动作。

### Policy Iteration

两步循环：

```text
Policy Evaluation: 固定 pi，算 V^pi
Policy Improvement: 根据 V^pi 改进 pi
```

直觉：先评估当前走法，再改进走法。

PI 不一定更快：

```text
PI: 外层轮数少，但每轮评估重。
VI: 每轮轻，但可能迭代多。
```

## 5. Cliff Walking: SARSA vs Q-Learning

两者都学 `Q(s,a)`，区别只在 TD target。

SARSA：

```text
r + gamma Q(s', a')
```

用实际采样到的 `a'`，所以是 on-policy。

Q-Learning：

```text
r + gamma max_a Q(s', a)
```

用下一状态的最优动作，所以是 off-policy。

Cliff Walking 里的现象：

```text
Q-Learning: 学理想 greedy 策略，贴悬崖，路径短但冒险。
SARSA: 把探索时可能掉悬崖的风险也算进去，路径更保守。
```

## 6. 探索

不探索会过早相信当前看起来不错的动作，错过更好策略。

epsilon-greedy：

```text
1 - epsilon: 选当前最优动作
epsilon: 随机探索
```

探索本质：

```text
用短期成本换长期信息。
```

但探索衰减太快也危险，尤其是 DQN：策略太早贪心，会让自己收集到的数据变窄。

## 7. TD、MC、lambda

```text
TD(0): 看一步，然后 bootstrap。
TD(n): 看 n 步，然后 bootstrap。
Monte Carlo: 看到 episode 结束，不 bootstrap。
```

容易混淆：

```text
TD(1) 如果指 n=1，不是 Monte Carlo。
TD(lambda=1) 在 episodic 任务里接近 Monte Carlo。
```

lambda：

```text
小 lambda: 更像 TD(0)，方差低，传播慢。
大 lambda: 更像 MC，传播快，方差高。
```

## 8. DQN

CartPole 状态连续，不能用 Q-table，所以 DQN 用网络近似：

```text
Q(s,a)
```

DQN 输出：

```text
Q(left), Q(right)
```

动作来自：

```text
argmax_a Q(s,a)
```

关键稳定器：

```text
Replay Buffer:
  随机抽旧经验，打破连续样本相关性。

Target Network:
  用 online network 的延迟拷贝算 target，避免目标每步漂移。
```

为什么表格 Q-Learning 不需要 target network？

```text
Q-table 每次只改一个格子；
DQN 一次梯度更新会影响很多状态的 Q 估计。
```

DQN 不是 Actor-Critic：

```text
online Q-network 和 target Q-network 都估计 Q(s,a)。
target network 只是延迟拷贝，不是 actor。
```

## 9. PPO

PPO 属于 Policy Gradient / Actor-Critic：

```text
REINFORCE -> REINFORCE + baseline -> Actor-Critic -> PPO
```

核心变化：

```text
DQN 学 Q(s,a)，再 argmax。
PPO 直接学 pi(a|s)，也就是动作概率。
```

CartPole 中：

```text
DQN: 输出 Q(left), Q(right)
PPO: actor 输出 P(left), P(right)，critic 输出 V(s)
```

概念桥：

```text
REINFORCE:
  用完整回报 G_t 更新策略，简单但方差高。

REINFORCE + baseline:
  用 G_t - V(s) 降低方差。

Actor-Critic:
  actor 选动作，critic 估价值。
```

PPO 的核心：

```text
用 advantage 判断动作比预期好还是差。
用 clipping 限制策略每次别改太猛。
```

ratio：

```text
ratio = pi_new(a|s) / pi_old(a|s)
```

PPO 不用 DQN 那种长期 replay buffer，因为它是 on-policy。旧数据太旧时，`pi_old` 和 `pi_new` 差太多，ratio 就失去意义。

## 10. 怎么读 PPO 代码

按这 4 块读 `ppo_agent.py`：

```text
ActorCritic:
  shared trunk + actor head + critic head

act():
  采样动作，记录 old_log_prob 和 V(s)

RolloutBuffer:
  存一批 on-policy rollout，用完清掉

update():
  advantage -> ratio -> clip -> 更新 actor-critic
```

先别纠结：

```text
GAE、entropy、value_coef、max_grad_norm、mini-batch
```

先抓住：

```text
actor 选动作；
critic 估价值；
advantage 判断动作好坏；
clip 限制策略更新幅度。
```

## 11. 深度 RL 重点看稳定性

不要只问：

```text
有没有跑出过高分？
```

要看：

```text
greedy eval 是否稳定
eval min 是否也高
loss 是否爆炸
多个 seed 是否可靠
ablation 后是否合理退化
```

DQN 重点看：

```text
Replay Buffer / Target Network / epsilon decay
```

PPO 重点看：

```text
KL / ClipFrac / EvalAvg / EvalMin
```

一句话：

```text
表格 RL 主要看算法逻辑；
深度 RL 还必须看训练系统稳不稳。
```

## 12. 推荐阅读顺序

```text
1. frozen_lake/diagnose.py
2. frozen_lake/env.py
3. frozen_lake/agent.py

4. cliff_walking/diagnose.py
5. cliff_walking/agent.py

6. cartpole/diagnose.py
7. cartpole/agent.py

8. cartpole/train_ppo.py
9. cartpole/ppo_agent.py
```

最终目标：

```text
当 AI 给你一段 RL 代码时，
你能判断它是在正确学习，
还是只是“看起来能跑”。
```
