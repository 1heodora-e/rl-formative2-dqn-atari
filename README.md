# rl-formative2-dqn-atari

**Formative 2 — DQN Atari (Stable-Baselines3 + Gymnasium)**

## Team Members
- **Honorine** — `ALE/Pong-v5`
- **Aubert** — `BreakoutNoFrameskip-v4`

---

# Honorine's Work — Pong

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

---

# Aubert's Work — Breakout

| Artifact | Path |
|----------|------|
| Training | `Aubert-Breakout/train.py` |
| Evaluation / video | `Aubert-Breakout/play.py` |
| Best model | `Aubert-Breakout/models/dqn_model.zip` |
| Results | `Aubert-Breakout/hyperparameter_results.csv` |
| Gameplay video | `Aubert-Breakout/breakout_dqn-Agent.mp4` |

---

## Setup

```bash
pip install "stable-baselines3[extra]" gymnasium[atari] ale-py autorom imageio pandas matplotlib
AutoROM --accept-license
```

---

## Train & play

```bash
python Aubert-Breakout/train.py
python Aubert-Breakout/play.py
```

**Greedy evaluation (rubric):** `play.py` uses `model.predict(..., deterministic=True)` for greedy action selection.

---

## Hyperparameter experiments (Aubert) — 10 runs + **Noted behavior**

*All experiments ran for 200k timesteps. Best model (exp01_baseline) was retrained with 500k timesteps for final submission.*

| Member | Exp | Key change | Hyperparameters (summary) | Eval Reward | Noted behavior (presentation) |
|--------|-----|------------|---------------------------|-------------|------------------------------|
| Aubert | **01** | **Baseline (Best)** | CNN, lr=1e-4, γ=0.99, batch=32, ε: 1.0→0.05 (0.1) | **35.6 ± 1.85** | **Best performance.** Balanced exploration-exploitation with stable learning. Agent learned to break bricks effectively. |
| Aubert | **02** | **High LR** | CNN, lr=5e-4, γ=0.99, batch=32, ε: 1.0→0.05 (0.1) | **0.0 ± 0.0** | **Complete failure.** Learning rate too high caused instability and divergence. Agent failed to learn any meaningful policy. |
| Aubert | **03** | **Low LR** | CNN, lr=5e-5, γ=0.99, batch=32, ε: 1.0→0.05 (0.1) | **13.0 ± 4.15** | **Slow learner.** More stable but needs significantly more timesteps to converge. Underfitting within 200k steps. |
| Aubert | **04** | **Lower γ (0.95)** | CNN, lr=1e-4, γ=0.95, batch=32, ε: 1.0→0.05 (0.1) | **16.4 ± 5.50** | **Short-sighted planning.** Agent focused on immediate rewards, less effective at long brick-breaking sequences. |
| Aubert | **05** | **Very low γ (0.90)** | CNN, lr=1e-4, γ=0.90, batch=32, ε: 1.0→0.05 (0.1) | **2.6 ± 1.96** | **Too myopic.** Severe performance degradation. Agent ignored future rewards, making poor strategic decisions. |
| Aubert | **06** | **Larger batch (64)** | CNN, lr=1e-4, γ=0.99, batch=64, ε: 1.0→0.05 (0.1) | **29.8 ± 8.35** | **Good but high variance.** More stable gradients but slower updates. Performance close to baseline with more variability. |
| Aubert | **07** | **Extended exploration** | CNN, lr=1e-4, γ=0.99, batch=32, ε: 1.0→0.02 (0.3) | **26.2 ± 11.27** | **Prolonged exploration phase.** Good performance but very high variance. Agent explored longer, delaying exploitation. |
| Aubert | **08** | **Fast ε decay** | CNN, lr=1e-4, γ=0.99, batch=32, ε: 1.0→0.1 (0.05) | **30.2 ± 4.79** | **Quick convergence.** Strong performance with early exploitation. Low variance indicates consistent strategy. |
| Aubert | **09** | **Tuned ε** | CNN, lr=1e-4, γ=0.99, batch=32, ε: 1.0→0.01 (0.15) | **26.0 ± 4.10** | **Balanced approach.** Good exploration-exploitation trade-off with consistent results. |
| Aubert | **10** | **MLP policy** | MLP, lr=1e-4, γ=0.99, batch=32, ε: 1.0→0.05 (0.1) | **0.0 ± 0.0** | **Architecture mismatch.** MLP cannot effectively process raw pixel observations for Atari. CNN is essential for visual games. |

**Takeaway:** **exp01_baseline (500k, CNN)** is the main submission model; **exp02 & exp10** demonstrate the importance of proper hyperparameter selection and CNN architecture for Atari games.

---

## Hyperparameter discussion (Aubert)

### What helped performance:
- **Moderate learning rate (1e-4):** Balanced learning speed with stability (exp01 vs exp02/exp03)
- **High gamma (0.99):** Enabled long-term planning crucial for brick-breaking sequences (exp01 vs exp04/exp05)
- **CNN policy:** Essential for processing visual input from raw pixels (exp01 vs exp10)
- **Balanced exploration:** Neither too fast nor too slow epsilon decay (exp01/exp08 vs exp07)

### What hurt performance:
- **Too high learning rate (5e-4):** Caused complete divergence and training instability (exp02)
- **Very low gamma (0.90):** Made agent myopic, ignoring future rewards (exp05)
- **MLP policy:** Fundamentally incompatible with image-based observations (exp10)
- **Extended exploration:** Delayed convergence and increased variance (exp07)

### Key trade-offs:
- **Learning rate:** Speed vs stability—too high diverges, too low underfits
- **Gamma (discount factor):** Short-term vs long-term planning horizon
- **Batch size:** Gradient stability vs update frequency (32 vs 64)
- **Epsilon schedule:** Exploration vs exploitation timing—impacts convergence speed and final performance
- **Policy architecture:** CNN required for spatial features in visual games; MLP only for low-dimensional state spaces

### Video demonstration:
See `Aubert-Breakout/breakout_dqn-Agent.mp4` for gameplay footage showing the trained agent breaking bricks in Breakout.

---
