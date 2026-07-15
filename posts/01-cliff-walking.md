# 同一座悬崖，两种走法——Cliff Walking 里的 Q-Learning 与 SARSA

## 场景

你是一个小人，站在 12×4 格子的左下角。终点在右下角。每一步扣 1 分，所以你想快点儿到。但你和终点之间的底部一整排都是悬崖——踩上去扣 100 分，直接弹回起点。

你会怎么走：**贴着悬崖抄近路，还是绕远路保安全？**

```
🏠 ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜
⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜
⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜
🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🌊 🏆
```

## 对决

我们把 Q-Learning 和 SARSA 放到同一个悬崖上，同样训练 500 轮。训练完，关掉探索（纯 greedy 模式），让它们各自走一遍：

```
Q-Learning（贴悬崖抄近路）：        SARSA（绕远路保安全）：

→  →  →  →  →  →  →  →  →  →  →  ↓    →  →  →  →  →  →  →  ↓
↑                                   ↓    ↑                    →  ↓
↑  X  X  X  X  X  X  X  X  X  X  G    ↑  X  X  X  X  X  X  X  →  G
S  →  →  →  →  →  →  →  →  →  →  →    S  ↑  ←  ←  ←  ←  ←  ←  ←
```

同一个环境，同一套奖励，同样的超参数。**路径完全不同。** 为什么？

## 追到那一行代码

两个 agent 的结构几乎一模一样——都用 epsilon-greedy 选动作，都用 Q-table 存状态价值。区别只在一个函数里的一行：

```python
# Q-Learning.update() —— 只看最优
best_next_action = np.argmax(self.Q[ny, nx, :])
td_target = reward + self.gamma * self.Q[ny, nx, best_next_action] * (not done)
#                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                          用下一状态的最大 Q 值算 target

# SARSA.update() —— 看实际选了啥
td_target = reward + self.gamma * self.Q[ny, nx, next_action] * (not done)
#                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                          用下一状态实际选的动作算 target
```

写成人话就是：

> **Q-Learning** 在算 target 时假设"我下一步会做最优选择"——即使训练时它自己正在随机乱走，它也假装没看见。它在学一个**理想中的最优策略**。
>
> **SARSA** 在算 target 时老老实实问"按我现在的走法（包括偶尔随机乱走），下一步值多少"。它把**探索时的风险也算了进去**。

## 为什么 SARSA 不敢贴悬崖？

训练时，两个 agent 都有 10% 的概率随机走。Q-Learning 在悬崖边随机踩了一脚掉下去，扣了 100 分——但它更新 Q 值的时候，只关心"如果我不乱走，这里值多少"，所以悬崖边的 Q 值仍然很高。它坚信自己 greedy 的时候不会掉下去。

SARSA 不一样。它在悬崖边随机踩了一脚掉下去，更新 Q 值时把它算进去了——"按照我现在的走路风格，站在这里就是有风险"。于是悬崖边的 Q 值被拉低了，最终学到的策略自觉绕开了悬崖。

**Q-Learning 学的是最优策略（off-policy），SARSA 学的是当前策略（on-policy）。**

## 跑跑看

```bash
python cliff_walking/diagnose.py
```

输出会直接告诉你两件事：
- Q-Learning 的 greedy 路径贴悬崖走了几步（edge-adjacent steps）
- SARSA 训练期间的**总分反而更高**——因为它在训练时掉了更少次悬崖

可以试试把 `epsilon` 从 0.1 改到 0.3，看看 SARSA 绕得更远；改成 0.01，看看 Q-Learning 的优势变大。

## 在这条路上的位置

Cliff Walking 用的是 **TD Learning**：不需要知道环境的转移概率，只需要真正采样到 `(s, a, r, s')` 就能学。这叫 **model-free**。

但它是表格方法——12×4 的格子，每个格子存 4 个 Q 值，刚好装得下。

下一篇我们反过来问：**如果转移概率已知，还需要采样吗？** → Frozen Lake。
