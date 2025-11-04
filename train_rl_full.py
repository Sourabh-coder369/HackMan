"""
Fast RL Training on Full Corpus
================================
Train RL meta-learner on entire corpus with optimized evaluation.
"""

import pickle
import numpy as np
from tqdm import tqdm
from rl_meta_learner import RLEnsembleAgent, RLMetaLearner
from hmm_model import EnhancedHMMPredictor
from ngram_model import NgramModel
from environment import HangmanEnvironment


def load_models():
    """Load pre-trained HMM and N-gram models"""
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
    
    print("✅ Models loaded")
    return hmm_predictor, ngram_model


def train_episode(agent, word, env):
    """Train on one game episode"""
    env.reset(word)
    agent.reset()
    done = False
    
    while not done:
        game_state = env.get_state()
        letter, _ = agent.get_action(game_state, eval_mode=False)
        next_state, reward, done, info = env.step(letter)
        correct = info.get('correct_guess', False)
        agent.update(letter, correct, next_state, done)
        loss = agent.meta_learner.train_step()
    
    return agent.episode_reward, env.won


def quick_eval(agent, words, num_games=50):
    """Quick evaluation on subset"""
    wins = 0
    for word in words[:num_games]:
        env = HangmanEnvironment([word])
        env.reset(word)
        agent.reset()
        done = False
        
        while not done:
            game_state = env.get_state()
            letter, _ = agent.get_action(game_state, eval_mode=True)
            next_state, reward, done, info = env.step(letter)
            agent.update(letter, info.get('correct_guess', False), next_state, done)
        
        if env.won:
            wins += 1
    
    return wins / num_games * 100


def main():
    """Main training loop"""
    print("=" * 70)
    print("🎮 RL META-LEARNER TRAINING - FULL CORPUS")
    print("=" * 70)
    
    # Load models and ALL data
    hmm_predictor, ngram_model = load_models()
    
    print("\nLoading FULL training corpus...")
    with open('Data/corpus.txt', 'r') as f:
        all_words = [line.strip().lower() for line in f if line.strip()]
    
    print(f"✅ Loaded {len(all_words):,} words\n")
    
    # Split train/val
    split_idx = int(len(all_words) * 0.95)
    train_set = all_words[:split_idx]
    val_set = all_words[split_idx:]
    
    print(f"Training set: {len(train_set):,} words")
    print(f"Validation set: {len(val_set):,} words")
    
    # Initialize RL agent
    meta_learner = RLMetaLearner(
        state_dim=10,
        num_actions=5,
        learning_rate=0.001
    )
    agent = RLEnsembleAgent(hmm_predictor, ngram_model, meta_learner)
    
    # Training parameters
    num_episodes = 10000  # More episodes for full corpus
    target_update_freq = 200
    eval_freq = 1000
    
    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"  Episodes: {num_episodes:,}")
    print(f"  Training words: {len(train_set):,}")
    print(f"  Validation words: {len(val_set):,}")
    print(f"  State dim: {meta_learner.state_dim}")
    print(f"  Actions: {meta_learner.num_actions}")
    print(f"{'='*70}\n")
    
    best_win_rate = 0
    episode_rewards = []
    episode_wins = []
    
    # Training loop
    print("Starting training...")
    pbar = tqdm(range(num_episodes), desc="Training")
    
    for episode in pbar:
        # Sample random word from FULL training set
        word = np.random.choice(train_set)
        env = HangmanEnvironment([word])
        
        # Train episode
        episode_reward, won = train_episode(agent, word, env)
        episode_rewards.append(episode_reward)
        episode_wins.append(won)
        
        # Update target network
        if episode % target_update_freq == 0:
            meta_learner.update_target_network()
        
        # Decay epsilon
        meta_learner.update_epsilon()
        
        # Update progress bar
        if episode % 50 == 0:
            recent_wr = np.mean(episode_wins[-200:]) * 100
            recent_reward = np.mean(episode_rewards[-200:])
            pbar.set_postfix({
                'epsilon': f'{meta_learner.epsilon:.3f}',
                'wr': f'{recent_wr:.1f}%',
                'reward': f'{recent_reward:.1f}'
            })
        
        # Quick evaluation (fast subset eval)
        if (episode + 1) % eval_freq == 0:
            print(f"\n{'='*70}")
            print(f"Quick Evaluation at episode {episode + 1}")
            print(f"{'='*70}")
            
            val_wr = quick_eval(agent, val_set, num_games=100)
            print(f"Validation (100 words): {val_wr:.1f}% win rate")
            print(f"Training (last 200): {np.mean(episode_wins[-200:])*100:.1f}% win rate")
            print(f"Epsilon: {meta_learner.epsilon:.3f}")
            
            # Save best model
            if val_wr > best_win_rate:
                best_win_rate = val_wr
                meta_learner.save('models/rl_meta_learner_best.pth')
                print(f"✅ New best model saved! Win rate: {val_wr:.1f}%")
            
            print(f"{'='*70}\n")
    
    # Save final model
    meta_learner.save('models/rl_meta_learner_final.pth')
    
    print("\n" + "="*70)
    print("✅ Training complete!")
    print(f"Best validation win rate: {best_win_rate:.1f}%")
    print(f"Models saved to:")
    print(f"  - models/rl_meta_learner_best.pth")
    print(f"  - models/rl_meta_learner_final.pth")
    print("\nNow run: python evaluate_rl_full.py")
    print("="*70)


if __name__ == '__main__':
    main()
