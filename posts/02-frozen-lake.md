# 同一片冰面，两种算法——Frozen Lake 里的 Value Iteration 与 Policy Iteration

## 场景

4×4 的冰面。左上角出发，右下角是终点，中间散落着冰窟窿。往上、下、左、右四个方向走——但冰面很滑。你选"往右"，真实结果可能是 80% 往右、10% 往上、10% 往下。

和上一篇悬崖不同：**这一次，你知道冰面的精确滑动规则。** 知道每一步踩下去所有可能的后果是什么。

那还试什么错？直接在家算好最优路线再出门。

## 对决

我们把 Value Iteration 和 Policy Iteration 放到同一片冰面上，用同一份转移模型。两种方法最终**收敛到同一个最优策略**——但到达的过程截然不同：

```
Value Iteration：                    Policy Iteration：
Bellman 最优扫描 23 轮               策略改进 5 轮
直接完工                             总评估扫描 183 轮
```

一个扫了 23 轮就收工，一个来回折腾了 183 轮。为什么？

## 追到核心差异

两种方法都在用 `get_transitions(state, action)` ——已知环境里所有可能的结果和概率，直接**算期望**而非采样。

区别在于怎么用这个模型：

```python
# Value Iteration —— 每轮直接用 max，一步到位
best_value = np.max(action_values)
new_V[y, x] = best_value
# V(s) = max_a Σ P(s'|s,a) [R + γ·V(s')]

# Policy Iteration —— 拆成两步
# Step 1 evaluate_policy(): 固定当前策略，老老实实算 V^π
value = expected_action_value(env, V, state, self.policy[y, x], gamma)
# Step 2 improve_policy(): 根据 V^π 改进策略
best_action = np.argmax(action_values)
self.policy[y, x] = best_action
```

写成人话：

> **Value Iteration** 是个急性子。每轮更新都直接取 `max`——"我不管当前策略是什么，我直接往最优的方向冲。"23 轮，一步到位。
>
> **Policy Iteration** 是个谨慎的棋手。先完整评估完当前走法的好坏（有时一次评估就要扫 100+ 轮），确认无误了，再改棋路。每次改棋路都深思熟虑，所以只改了 5 次。但每次改动前的评估很花时间。

## VI 和 PI：两条路，同一个终点

最终两种方法的价值函数差异几乎为零，策略完全一致。这说明：

> 到达最优策略可以有不同的路径——可以"一直朝最优冲"（VI），也可以"评估一轮、改一轮、再评估"（PI）。

而这个"评估→改进"的循环，正是 RL 里最核心的框架——**Generalized Policy Iteration**。后面所有算法本质上都是它的近似版本：

```
Actor-Critic = Critic 做评估 + Actor 做改进
Q-Learning   = 用采样和 TD target 近似这套循环
```

Policy Iteration 代码虽然比 VI 复杂，但它揭示了 RL 的底层骨架。

## 跑跑看

```bash
python frozen_lake/diagnose.py
```

输出告诉你两件事：
- VI 扫了多少轮、PI 改了几次策略
- 最终两种方法的价值函数差异和策略差异（应该几乎为零）

可以改 `slip_prob`（冰面滑的概率）看看——滑率越高，策略越保守。

## 在这条路上的位置

Frozen Lake 用的是 **Dynamic Programming**——需要完整环境模型 `P(s'|s,a)`，做的是 **full backup**（枚举所有可能结果）。

上一篇 Cliff Walking 是 model-free（靠采样），这一篇是 model-based（靠计算）。但它们的共同前提是——状态数有限，装得进表格。

下一篇：**状态多到表格装不下，怎么办？** → CartPole。
