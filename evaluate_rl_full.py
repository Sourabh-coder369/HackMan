"""
Full RL Meta-Learner Evaluation
================================
Test RL agent on ENTIRE test set (all 2000 words).
"""

import pickle
import numpy as np
from tqdm import tqdm
from rl_meta_learner import RLEnsembleAgent, RLMetaLearner
from hmm_model import EnhancedHMMPredictor
from ngram_model import NgramModel
from environment import HangmanEnvironment


def load_models():
    """Load all models"""
    print("Loading models...")
    
    # Load corpus words
    with open('Data/corpus.txt', 'r') as f:
        corpus_words = [line.strip().lower() for line in f if line.strip()]
    
    # Load HMM
    with open('models/hmm_collection.pkl', 'rb') as f:
        hmm_collection = pickle.load(f)
    hmm_predictor = EnhancedHMMPredictor(hmm_collection, corpus_words)
    
    # Load N-gram
    ngram_model = NgramModel.load('models/ngram_model.pkl')
    
    # Load RL meta-learner
    meta_learner = RLMetaLearner()
    meta_learner.load('models/rl_meta_learner_best.pth')
    
    print("✅ Models loaded\n")
    return hmm_predictor, ngram_model, meta_learner


print("=" * 70)
print("🎯 FULL RL META-LEARNER EVALUATION (ALL TEST WORDS)")
print("=" * 70)

# Load models
hmm_predictor, ngram_model, meta_learner = load_models()

# Load ENTIRE test set
with open('Data/test.txt', 'r') as f:
    test_words = [line.strip().lower() for line in f if line.strip()]

print(f"Test set: {len(test_words)} words\n")

# Test RL agent on ALL words
print("Testing RL-Enhanced Ensemble on FULL test set...")
rl_agent = RLEnsembleAgent(hmm_predictor, ngram_model, meta_learner)

rl_wins = 0
rl_score = 0
rl_wrong = 0
rl_repeated = 0
action_counts = {i: 0 for i in range(5)}

for word in tqdm(test_words, desc="RL Ensemble"):
    env = HangmanEnvironment([word])
    env.reset(word)
    rl_agent.reset()
    done = False
    
    while not done:
        game_state = env.get_state()
        letter, debug = rl_agent.get_action(game_state, eval_mode=True)
        action_counts[debug['action']] += 1
        next_state, reward, done, info = env.step(letter)
        rl_agent.update(letter, info.get('correct_guess', False), next_state, done)
    
    if env.won:
        rl_wins += 1
        rl_score += 100
    rl_score += -5 * env.wrong_guesses - 2 * env.repeated_guesses
    rl_wrong += env.wrong_guesses
    rl_repeated += env.repeated_guesses

rl_wr = rl_wins / len(test_words) * 100
rl_avg = rl_score / len(test_words)

# Test baseline on ALL words
print("\nTesting Fixed-Weight Baseline (40% HMM, 60% N-gram)...")
baseline_wins = 0
baseline_score = 0
baseline_wrong = 0
baseline_repeated = 0

for word in tqdm(test_words, desc="Baseline"):
    env = HangmanEnvironment([word])
    env.reset(word)
    guessed = set()
    done = False
    
    while not done:
        game_state = env.get_state()
        masked_word = game_state['masked_word']
        
        revealed = set(c for c in masked_word if c != '_' and c != ' ')
        wrong_letters = list(guessed - revealed)
        guessed_set = set(g.upper() for g in guessed)
        
        hmm_probs = hmm_predictor.predict(masked_word, guessed_set, wrong_letters)
        ngram_probs = ngram_model.predict_letter_probs(masked_word, list(guessed))
        
        best_letter = None
        best_score = -1
        available = set('abcdefghijklmnopqrstuvwxyz') - guessed
        
        for letter in available:
            hmm_s = hmm_probs.get(letter.upper(), 0)
            ngram_s = ngram_probs[ord(letter) - ord('a')]
            score = 0.4 * hmm_s + 0.6 * ngram_s
            if score > best_score:
                best_score = score
                best_letter = letter
        
        guessed.add(best_letter)
        next_state, reward, done, info = env.step(best_letter)
    
    if env.won:
        baseline_wins += 1
        baseline_score += 100
    baseline_score += -5 * env.wrong_guesses - 2 * env.repeated_guesses
    baseline_wrong += env.wrong_guesses
    baseline_repeated += env.repeated_guesses

baseline_wr = baseline_wins / len(test_words) * 100
baseline_avg = baseline_score / len(test_words)

# Results
print("\n" + "=" * 70)
print(f"📊 FULL TEST SET RESULTS ({len(test_words)} words)")
print("=" * 70)

print(f"\n{'Metric':<30} {'Baseline':<18} {'RL Enhanced':<18} {'Δ':<12}")
print("-" * 70)

print(f"{'Win Rate':<30} {baseline_wr:>8.2f}%{'':<9} {rl_wr:>8.2f}%{'':<9} {rl_wr - baseline_wr:>+8.2f}%")
print(f"{'Wins':<30} {baseline_wins:>8}{'':<10} {rl_wins:>8}{'':<10} {rl_wins - baseline_wins:>+8}")
print(f"{'Total Score':<30} {baseline_score:>8}{'':<10} {rl_score:>8}{'':<10} {rl_score - baseline_score:>+8}")
print(f"{'Avg Score':<30} {baseline_avg:>8.1f}{'':<10} {rl_avg:>8.1f}{'':<10} {rl_avg - baseline_avg:>+8.1f}")
print(f"{'Wrong Guesses':<30} {baseline_wrong:>8}{'':<10} {rl_wrong:>8}{'':<10} {rl_wrong - baseline_wrong:>+8}")
print(f"{'Repeated Guesses':<30} {baseline_repeated:>8}{'':<10} {rl_repeated:>8}{'':<10} {rl_repeated - baseline_repeated:>+8}")

# Action distribution
print("\n" + "=" * 70)
print("🎮 RL ACTION DISTRIBUTION")
print("=" * 70)

total_actions = sum(action_counts.values())
print(f"\nTotal decisions: {total_actions:,}\n")

for action in range(5):
    count = action_counts[action]
    pct = count / total_actions * 100 if total_actions > 0 else 0
    hmm_w, ngram_w = meta_learner.action_to_weights[action]
    bar = '█' * int(pct / 2)
    print(f"Action {action} (HMM:{hmm_w:4.0%}, N-gram:{ngram_w:4.0%}): {count:6,} ({pct:5.1f}%) {bar}")

# Summary
print("\n" + "=" * 70)
print("🏆 FINAL VERDICT")
print("=" * 70)

improvement = rl_wr - baseline_wr

if improvement > 0.5:
    print(f"\n✅ RL META-LEARNER WINS!")
    print(f"   Improvement: +{improvement:.2f}% win rate")
    print(f"   {baseline_wr:.2f}% → {rl_wr:.2f}%")
    print(f"   {rl_wins - baseline_wins:+d} more wins on {len(test_words)} words")
    print(f"\n   🏆 Best Model: RL Meta-Learner")
    print(f"   Files: models/rl_meta_learner_best.pth")
elif improvement > -0.5:
    print(f"\n⚖️  TIE: RL matches baseline performance!")
    print(f"   RL: {rl_wr:.2f}% | Baseline: {baseline_wr:.2f}%")
    print(f"   Difference: {improvement:+.2f}%")
    print(f"\n   Both models are equally good!")
    print(f"   ✅ Use RL version to showcase advanced ML skills")
else:
    print(f"\n➡️  Baseline edges ahead")
    print(f"   Baseline: {baseline_wr:.2f}% | RL: {rl_wr:.2f}%")
    print(f"   Difference: {improvement:+.2f}%")
    print(f"\n   BUT: You still have a working RL implementation!")

print("\n" + "=" * 70)
print("💡 KEY INSIGHTS")
print("=" * 70)

# Analyze which actions RL prefers
preferred_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:3]
print(f"\nRL Agent's Top 3 Preferred Actions:")
for i, (action, count) in enumerate(preferred_actions, 1):
    pct = count / total_actions * 100
    hmm_w, ngram_w = meta_learner.action_to_weights[action]
    print(f"  {i}. Action {action} (HMM:{hmm_w:.0%}, N-gram:{ngram_w:.0%}): {pct:.1f}% of time")

print(f"\n📈 Performance per Win:")
print(f"   Baseline: {baseline_score / max(baseline_wins, 1):.1f} avg score per win")
print(f"   RL Agent: {rl_score / max(rl_wins, 1):.1f} avg score per win")

print("\n" + "=" * 70)
