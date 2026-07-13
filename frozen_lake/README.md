# Frozen Lake: Dynamic Programming

这个目录用 Frozen Lake 讲动态规划方法：Value Iteration 和 Policy Iteration。

它和 Cliff Walking 的 SARSA / Q-Learning 最大区别是：这里我们假设已经知道完整环境模型，所以可以直接规划，而不是靠采样试错。

## 先看 `get_transitions()`

核心入口在 `env.py`：

```python
get_transitions(state, action)
```

它不是“真的走一步”，而是枚举：

```text
在状态 s 选择动作 a 后，所有可能发生的结果及其概率。
```

也就是 DP 需要的模型：

```text
P(s', r | s, a)
```

注意这里不只是 `s`，还包括 `a`。

同一个格子里，选择不同动作，会对应不同的转移概率分布。例如动作编号是：

```text
0: Up
1: Right
2: Down
3: Left
```

如果你选择 `Right`，但冰面会滑，那么真实结果可能是：

```text
90% Right
5% Up
5% Down
```

如果你选择 `Up`，真实结果又会变成：

```text
90% Up
5% Left
5% Right
```

所以转移模型是：

```text
P(s' | s, a)
```

不是：

```text
P(s' | s)
```

可以这样记：

```text
state 决定你在哪里；
action 决定你试图往哪里走；
transition model 决定这个尝试会以哪些概率变成真实结果。
```

## DP 在算什么

因为我们知道每个动作的所有可能结果，所以可以直接算期望：

```text
Q(s,a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')]
```

也就是：

```text
这个动作的价值 =
所有可能结果的概率 * 结果价值
再加总
```

这叫 full backup。

和 TD 方法对比：

```text
DP: 知道完整模型，枚举所有 s'，算期望
TD: 不知道完整模型，只用真实采样到的一步 s,a,r,s'
```

## Value Iteration

Value Iteration 直接逼近最优价值函数：

```text
V(s) <- max_a sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')]
```

它每次更新都直接取 `max_a`。

直觉是：

```text
如果我下一步永远选当前看起来最好的动作，这个状态值多少？
```

代码结构很简单：

```text
对每个 state:
  计算所有 action 的 Q(s,a)
  V(s) = max_a Q(s,a)
  policy(s) = argmax_a Q(s,a)
```

所以 Value Iteration 更像：

```text
直接朝最优价值函数 V* 推过去。
```

## Policy Iteration

Policy Iteration 看起来更复杂，因为它显式拆成两步：

```text
1. Policy Evaluation
   固定当前策略 pi，计算 V^pi

2. Policy Improvement
   根据 V^pi，对每个状态选择更好的动作
```

也就是：

```text
先问：按照当前这套走法，每个格子值多少？
再问：知道这些价值后，我要不要换动作？
```

外层循环是：

```text
评估当前策略 -> 改进策略 -> 再评估 -> 再改进
```

直到策略不再变化。

它比 Value Iteration 代码复杂，但它很重要，因为它揭示了强化学习里的一个核心框架：

```text
policy evaluation + policy improvement
```

后面的很多算法都可以看成这个框架的近似版本：

```text
Actor-Critic:
  Critic 做 policy evaluation
  Actor 做 policy improvement

SARSA / Q-Learning:
  用采样和 TD target 近似 evaluation / improvement

Generalized Policy Iteration:
  几乎所有控制型 RL 都在交替做“评估”和“改进”
```

所以学习建议是：

```text
先用 Value Iteration 理解 Bellman 最优方程；
再用 Policy Iteration 理解 evaluation / improvement 这套大框架。
```

## 怎么运行

动态演示 Value Iteration：

```bash
python3 rl_playground/frozen_lake/train_play.py --mode value
```

动态演示 Policy Iteration：

```bash
python3 rl_playground/frozen_lake/train_play.py --mode policy
```

在同一个入口里快速对比两种方法：

```bash
python3 rl_playground/frozen_lake/train_play.py --mode compare
```

运行 DP 对比诊断：

```bash
python3 rl_playground/frozen_lake/diagnose.py
```

你会看到类似：

```text
Value Iteration
- Bellman optimality sweeps: 23

Policy Iteration
- Policy improvement rounds: 5
- Total policy-evaluation sweeps: 183

Comparison
- Max |V_value_iteration - V_policy_iteration|: very small
- Greedy policy mismatches: 0/11
```

这说明两种方法虽然过程不同，但最终会收敛到同一个最优策略。

## 读代码顺序

1. `env.py`

   先看 `get_transitions()`。这是 DP 能工作的前提。

2. `agent.py`

   先看 `ValueIterationAgent.step_value_iteration()`，再看 `PolicyIterationAgent.evaluate_policy()` 和 `PolicyIterationAgent.improve_policy()`。

3. `diagnose.py`

   看它如何比较两种方法的收敛轮数、价值函数差异和最终策略。

一句话总结：

```text
Value Iteration 更像“直接求最优价值”；
Policy Iteration 更像“先评估当前策略，再逐步改好策略”。
```
