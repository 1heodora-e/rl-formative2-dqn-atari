# rl-formative2-dqn-atari

**Formative 2 — DQN Atari (Stable-Baselines3 + Gymnasium)**  
**Honorine — `ALE/Pong-v5`**

| Artifact | Path |
|----------|------|
| Training | `Honorine-pong/train.py` |
| Evaluation / video | `Honorine-pong/play.py` |
| Best model (typ.) | `Honorine-pong/dqn_model.zip` or `runs/exp01/cnn/dqn_model.zip` |
| Results | `runs/hyperparameter_results.csv` |

---

## Setup

```bash
pip install "stable-baselines3[extra]" gymnasium[atari] ale-py autorom imageio pandas matplotlib
AutoROM --accept-license
```

---

## Train & play

```bash
python Honorine-pong/train.py --env-id ALE/Pong-v5 --policy cnn --exp-name exp01 --total-timesteps 1000000
python Honorine-pong/play.py --env-id ALE/Pong-v5 --model-path dqn_model.zip --policy cnn --episodes 3 --render-mode human
```

Kaggle (video): add `--render-mode rgb_array --save-video --video-path pong_agent.mp4`.

**Greedy evaluation (rubric):** `play.py` uses `model.predict(..., deterministic=True)` (same idea as GreedyQPolicy).

---

## Hyperparameter experiments (Honorine) — 10 runs + **Noted behavior**

*ε decay = `exploration_fraction` in SB3. CNN except **exp03** (MLP).*

| Member | Exp | Key change | Hyperparameters (summary) | Noted behavior (presentation) |
|--------|-----|------------|-------------------------|------------------------------|
| Honorine | **01** | **1M steps** | CNN, lr=1e-4, γ=0.99, batch=32, ε default | **Best performer.** Agent tracks the ball well and scores frequently. |
| Honorine | **02** | **Lower LR** | CNN, lr=5e-5, … | **Slow learner.** Agent still “jittering”; missed fast balls. |
| Honorine | **03** | **MLP policy** | MlpPolicy, lr=1e-4, … | **Total failure.** Agent never learned to move toward the ball. |
| Honorine | 04 | Higher LR | CNN, lr=2e-4, … | Faster early change vs baseline; watch stability vs exp01. |
| Honorine | 05 | Lower γ | CNN, γ=0.95, … | More myopic credit assignment vs γ=0.99. |
| Honorine | 06 | Higher γ | CNN, γ=0.999, … | Stronger long-horizon signal; compare learning speed to exp01. |
| Honorine | 07 | Batch 64 | CNN, batch=64, … | Gradient noise vs batch 32 trade-off. |
| Honorine | **08** | **Batch 128** | CNN, batch=128, higher ε_end | **Stable learning**, but did **not** reach high scores as quickly as batch 32. |
| Honorine | 09 | 300k + ε schedule | CNN, 300k steps, ε_end=0.01, decay 0.20 | Shorter run + exploration schedule; compare final skill to exp01/exp10. |
| Honorine | **10** | **500k steps** | CNN, 500k, ε decay 0.05 | **Strong**, but **slightly less consistent** than 1M; still beats the computer often. |

**Takeaway:** **exp01 (1M, CNN)** is the main submission model; **exp03** documents why **CNN > MLP** on raw pixels for Pong.

---

## Hyperparameter discussion (short)

- **Helped:** More training steps (exp01 vs exp10), moderate lr + CNN spatial features.  
- **Hurt:** MLP on flattened frames (exp03), very low lr slowing progress (exp02), large batch slowing improvement vs 32 (exp08).  
- **Trade-offs:** batch size vs update noise; ε schedule vs late-game greed; γ vs planning horizon.
---
## Technical Note on Rendering:
To comply with the rubric's requirement for visualization while working in a headless Cloud environment (Kaggle), we utilized the Gymnasium video recorder wrapper. This renders the env.render() frames into an MP4 format, allowing for clear evaluation of the agent's behavior without requiring a local GUI.
