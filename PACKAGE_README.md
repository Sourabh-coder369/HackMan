# RL Meta-Learner Package - Complete File List
**What to send to your friend to run the RL Meta-Learner**

---

## Required Files

### 1. Core Python Files (REQUIRED)
```
rl_meta_learner.py          # Main RL agent implementation
hmm_model.py                # HMM predictor
ngram_model.py              # N-gram language model
environment.py              # Hangman game environment
```

### 2. Pre-trained Models (REQUIRED)
```
models/
├── hmm_collection.pkl          # Pre-trained HMM (50K words)
├── ngram_model.pkl             # Pre-trained N-gram (50K words)
├── rl_meta_learner_best.pth   # Trained RL agent weights
└── rl_meta_learner_final.pth  # Alternative RL checkpoint
```

### 3. Data Files (REQUIRED for evaluation)
```
Data/
├── corpus.txt              # Training corpus (50K words)
└── test.txt                # Test set (2K words)
```

### 4. Evaluation Script (REQUIRED)
```
evaluate_rl_full.py         # Full test set evaluation script
```

### 5. Dependencies (REQUIRED)
```
requirements.txt            # Python package dependencies
```

### 6. Documentation (OPTIONAL but recommended)
```
Analysis_Report_Compact.md  # Technical report (100 lines)
RL_QUICKSTART.md           # Quick start guide
README.md                  # Project overview
```

---

## Minimal Package (Just to Run)

If your friend just wants to **evaluate the trained model**, send:

### Essential Files Only:
1. `rl_meta_learner.py`
2. `hmm_model.py`
3. `ngram_model.py`
4. `environment.py`
5. `evaluate_rl_full.py`
6. `models/hmm_collection.pkl`
7. `models/ngram_model.pkl`
8. `models/rl_meta_learner_best.pth`
9. `Data/test.txt`
10. `requirements.txt`

**Total:** 10 files (~52 MB)

---

## Full Package (To Train + Evaluate)

If your friend wants to **train from scratch**, send:

### All Required Files:
1. Core Python files (4 files)
2. Pre-trained base models (2 files: HMM + N-gram)
3. Data files (2 files: corpus + test)
4. Training script: `train_rl_full.py`
5. Evaluation script: `evaluate_rl_full.py`
6. `requirements.txt`

**Total:** 12 files (~102 MB)

---

## File Sizes (Approximate)

```
rl_meta_learner.py          ~15 KB
hmm_model.py                ~10 KB
ngram_model.py              ~6 KB
environment.py              ~5 KB
evaluate_rl_full.py         ~5 KB
train_rl_full.py            ~6 KB

models/hmm_collection.pkl   ~25 MB
models/ngram_model.pkl      ~20 MB
models/rl_meta_learner_best.pth  ~50 KB

Data/corpus.txt             ~450 KB
Data/test.txt               ~20 KB

requirements.txt            ~1 KB
```

**Total Package Size:** ~52 MB (minimal) or ~102 MB (full)

---

## Installation & Usage Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- torch>=2.0.0
- numpy>=1.24.0

### Step 2: Run Evaluation
```bash
python evaluate_rl_full.py
```

**Expected Output:**
```
Loading models...
Testing on 2000 words...
Progress: 500/2000 | Win rate: 15.40%
Progress: 1000/2000 | Win rate: 15.20%
Progress: 1500/2000 | Win rate: 15.27%
Progress: 2000/2000 | Win rate: 15.30%

Final Results:
Win rate: 15.30% (306/2000 wins)
```

### Step 3 (Optional): Re-train Model
```bash
python train_rl_full.py
```

**Training time:** ~34 minutes on CPU

---

## Quick Test (Single Word)

Create a simple test script:

```python
# test_single_word.py
from rl_meta_learner import RLEnsembleAgent, RLMetaLearner
from hmm_model import EnhancedHMMPredictor
from ngram_model import NgramModel
import pickle
import torch

# Load models
with open('models/hmm_collection.pkl', 'rb') as f:
    hmm_collection = pickle.load(f)
with open('Data/corpus.txt', 'r') as f:
    corpus_words = [line.strip().upper() for line in f]

hmm = EnhancedHMMPredictor(hmm_collection, corpus_words)
ngram = NgramModel.load('models/ngram_model.pkl')

# Load RL agent
meta_learner = RLMetaLearner(hmm, ngram, state_dim=10, num_actions=5)
meta_learner.policy_net.load_state_dict(
    torch.load('models/rl_meta_learner_best.pth')
)

# Create agent
agent = RLEnsembleAgent(hmm, ngram, meta_learner)

# Test on a word
word = "PYTHON"
word_state = ['_'] * len(word)
guessed = []

print(f"Word: {word}")
for i in range(6):  # Max 6 guesses
    guess = agent.get_action(word_state, guessed)
    print(f"Guess {i+1}: {guess}")
    guessed.append(guess)
    
    if guess in word:
        for j, letter in enumerate(word):
            if letter == guess:
                word_state[j] = guess
    
    print(f"State: {''.join(word_state)}")
    if '_' not in word_state:
        print("✓ Won!")
        break
```

Run: `python test_single_word.py`

---

## Folder Structure

After extracting, folder should look like:

```
rl_meta_learner_package/
│
├── rl_meta_learner.py
├── hmm_model.py
├── ngram_model.py
├── environment.py
├── evaluate_rl_full.py
├── train_rl_full.py (optional)
├── requirements.txt
├── README.md (optional)
│
├── models/
│   ├── hmm_collection.pkl
│   ├── ngram_model.pkl
│   └── rl_meta_learner_best.pth
│
└── Data/
    ├── corpus.txt
    └── test.txt
```

---

## Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'torch'"
**Solution:** Install PyTorch: `pip install torch`

### Issue 2: "FileNotFoundError: [Errno 2] No such file or directory: 'models/...'"
**Solution:** Ensure folder structure is correct. Models must be in `models/` subfolder.

### Issue 3: "RuntimeError: Error(s) in loading state_dict"
**Solution:** Make sure rl_meta_learner.py matches the version used for training.

### Issue 4: Low performance / Different results
**Solution:** Ensure using `rl_meta_learner_best.pth` (not final). Results may vary ±1% due to randomness.

---

## Packaging Commands

### Create ZIP Package (Windows PowerShell):
```powershell
# Create package directory
New-Item -ItemType Directory -Path "rl_package"

# Copy core files
Copy-Item rl_meta_learner.py, hmm_model.py, ngram_model.py, environment.py, evaluate_rl_full.py, requirements.txt rl_package/

# Copy models
Copy-Item models -Recurse rl_package/

# Copy data
Copy-Item Data -Recurse rl_package/

# Copy documentation (optional)
Copy-Item Analysis_Report_Compact.md, README.md rl_package/

# Create ZIP
Compress-Archive -Path rl_package/* -DestinationPath rl_meta_learner_package.zip
```

### Create ZIP Package (Linux/Mac):
```bash
# Create package directory
mkdir rl_package

# Copy files
cp rl_meta_learner.py hmm_model.py ngram_model.py environment.py evaluate_rl_full.py requirements.txt rl_package/

# Copy folders
cp -r models Data rl_package/

# Optional: Copy docs
cp Analysis_Report_Compact.md README.md rl_package/

# Create tar.gz
tar -czf rl_meta_learner_package.tar.gz rl_package/
```

---

## Verification Checklist

Before sending, verify package contains:

- [ ] `rl_meta_learner.py` exists
- [ ] `hmm_model.py` exists
- [ ] `ngram_model.py` exists
- [ ] `environment.py` exists
- [ ] `evaluate_rl_full.py` exists
- [ ] `models/hmm_collection.pkl` exists (~25 MB)
- [ ] `models/ngram_model.pkl` exists (~20 MB)
- [ ] `models/rl_meta_learner_best.pth` exists (~50 KB)
- [ ] `Data/test.txt` exists (~20 KB)
- [ ] `requirements.txt` exists
- [ ] Total package size: ~52 MB

---

## Expected Performance

When your friend runs `evaluate_rl_full.py`, they should see:

```
✓ Win rate: 15.30% ± 0.5%
✓ Wins: 306 ± 10
✓ Action 3 usage: ~70%
✓ Action 1 usage: ~30%
✓ Evaluation time: ~20 seconds
```

Small variations (±1%) are normal due to randomness in action selection.

---

## Contact & Support

If your friend has issues:
1. Check Python version (3.8+)
2. Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
3. Ensure all files are in correct folders
4. Try the single-word test script first

---

**Package Created:** November 3, 2025  
**Version:** 1.0  
**Performance:** 15.30% win rate on 2K test words  
**Package Size:** ~52 MB (minimal) | ~102 MB (full)
