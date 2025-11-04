# ML Hackathon - Analysis Report

**Team Members:**
- Sourabh S M - PES1UG23AM313
- Subashri V - PES1UG23AM319
- Suhas Hasoor - PES1UG23AM322
- Thanushree CB - PES1UG23AM338

**Project:** RL Meta-Learner for Hangman  
**Date:** November 2025

---

## 1. Key Observations

### Challenges

**Challenge 1: State Design**
Encoding Hangman's dynamic state into a 10D vector while keeping it compact and meaningful was complex. We needed to capture partial word information, model confidence, agreement metrics, and game progress while keeping features normalized and meaningful for neural network training.

**Challenge 2: Reward Engineering**
Balancing immediate vs. terminal rewards was key. Our final approach combines:
- Immediate feedback: **+10 correct, -5 wrong, -2 repeat**
- Terminal goals: **+100 win, -50 loss**

This promotes efficient, strategic play by balancing short-term learning signals with long-term objectives.

**Challenge 3: Model Integration**
Merging HMM (dict outputs) and N-gram (arrays) required normalization and careful type handling to ensure proper blending of predictions.

### Insights

**Insight 1: Binary Switching Policy**
The RL agent adopted a **binary switching policy** instead of gradual blending:
- **Action 3 (75% HMM): 70.3%** 
- **Action 1 (25% HMM): 29.7%**

This emergent behavior suggests optimal policy is context-dependent switching, not smooth interpolation.

**Insight 2: Strong Domain Priors**
Simple heuristic ensembles rivaled RL performance. Fixed 40-60 ensemble (16.60%) slightly outperforms RL (15.30%), showing that domain heuristics are powerful for structured problems like Hangman.

**Insight 3: Replay Buffer Stabilization**
Replay buffer (10K transitions) stabilized training and prevented divergence. Without it, training was unstable with wild oscillations.

---

## 2. Strategies

### HMM Design
**Position-based HMM** trained on **50,000 words** with:
- Bigram transitions for sequential letter dependencies
- Candidate filtering to narrow down possible words
- Position-specific frequency tables (beginning, middle, end)

This approach improved positional accuracy by learning letter patterns at specific word positions.

### RL State & Reward Design

**10-Dimensional State:**
- **Progress, Lives, Entropy, Complexity** - Game state metrics
- **HMM/N-gram Confidence** - Individual model certainty  
- **Model Agreement, Top-K Overlap** - Consensus indicators
- **Game Phase, Guessed Ratio** - Strategic context

The state captures uncertainty, context, and model consensus for informed blending decisions.

**Reward Structure:**
```
Immediate: +10 correct, -5 wrong, -2 repeated
Terminal: +100 win, -50 loss
```

This reward combined **short-term feedback** (+10/–5) with **terminal goals** (+100/–50), promoting efficient, strategic play by balancing immediate learning signals with long-term objectives.

### RL Algorithm: Double DQN
- **Network:** 10 → 64 → 64 → 5 (2 hidden layers, ReLU)
- **Experience Replay:** 10K buffer, batch size 128
- **Target Network:** Soft updates (τ=0.001)
- **Optimizer:** Adam (lr=0.001)

Chosen for stable convergence, sample efficiency, and proven performance.

---

## 3. Exploration vs. Exploitation

### Epsilon-Greedy Strategy
Used **ε-greedy policy** with linear decay: **ε = 1.0 → 0.01** over **10,000 episodes**

**Training Phases:**
1. **Early phase (Episodes 0-3000):** Exploration-heavy (ε > 0.70)
   - Agent tries all actions uniformly
   - Discovers which states favor which blending strategies
   
2. **Mid-phase (Episodes 3000-7000):** Balanced learning (ε = 0.70→0.30)
   - Key learning occurs
   - Win rate climbs from 12% → 15%
   
3. **Final phase (Episodes 7000-10000):** Exploitation (ε < 0.30)
   - Mostly uses learned policy
   - Policy refinement and fine-tuning

**Replay Buffer:** 10,000-transition buffer ensured diverse experience and stable convergence, preventing the agent from forgetting successful strategies.

---

## 4. Future Improvements

**Advanced RL Techniques:**
- **Dueling or Prioritized DQN** for better value estimation
- **Distributional RL** to model Q-value distributions
- **Multi-step returns** for improved credit assignment

**State Enhancements:**
- **Letter embeddings** and **top-K attention** mechanisms
- **Sequence encoding** with LSTM/Transformer for pattern history
- Capture richer relationships between letters and game state

**Model Upgrades:**
- **Transformer/LSTM word models** or pre-trained language models (BERT/GPT-2)
- **Larger corpus** (expand from 50K to millions of words)
- **Better HMM design** with context-dependent probabilities

**Training Improvements:**
- **Curriculum learning** (start with easy words, increase difficulty)
- **Adaptive reward scaling** by word difficulty
- **Auxiliary tasks** and reward shaping for guided exploration

---

## 5. Results & Conclusion

**Performance Metrics:**
- **RL Agent Win Rate:** 15.3% (306/2000 wins)
- **Baseline Ensemble:** 16.6% (332/2000 wins)  
- **Training Time:** 10K episodes, 34 minutes

**Key Insights:**
- RL agent learned **adaptive blending** between HMM and N-gram models
- Discovered **binary switching policy** (primarily using 25% and 75% HMM weights)
- Stable convergence with **experience replay** and **target network**
- Performance trailed **heuristic baseline** by 1.3%

**Conclusion:**
The RL meta-learner successfully demonstrates **advanced machine learning techniques** including reinforcement learning, neural networks, and meta-learning. While it slightly underperformed the fixed baseline, it provides a **strong foundation** for future research in:
- **Hybrid RL systems** combining learned and heuristic strategies
- **Meta-learning** for adaptive model selection
- **Contextual decision-making** in structured prediction tasks

This project showcases sophisticated **technical implementation**, emergent strategy discovery, and provides valuable insights for future improvements in reinforcement learning for game-playing agents.

---

**Technical:** 368-line implementation | PyTorch + NumPy | 34-min training | 15.30% win rate
