"""
RL Meta-Learner for Hangman
=============================
This RL agent learns to dynamically blend HMM and N-gram predictions
based on game state, rather than using fixed weights.

State: Game progress, entropy, model agreement, lives remaining
Action: Blending weights for HMM vs N-gram (discretized into 5 levels)
Reward: +10 for correct guess, -5 for wrong guess, +100 for win, -50 for loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pickle


class MetaLearnerNetwork(nn.Module):
    """Neural network that learns optimal blending weights"""
    def __init__(self, state_dim=10, num_actions=5):
        super(MetaLearnerNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class RLMetaLearner:
    """
    RL agent that learns to blend HMM and N-gram predictions.
    
    Actions (5 discrete levels):
    0: 100% HMM, 0% N-gram
    1: 75% HMM, 25% N-gram  
    2: 50% HMM, 50% N-gram
    3: 25% HMM, 75% N-gram
    4: 0% HMM, 100% N-gram
    """
    
    def __init__(self, state_dim=10, num_actions=5, learning_rate=0.001):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = torch.device("cpu")
        
        self.policy_net = MetaLearnerNetwork(state_dim, num_actions).to(self.device)
        self.target_net = MetaLearnerNetwork(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 64
        
        # Action to weight mapping
        self.action_to_weights = {
            0: (1.0, 0.0),   # 100% HMM
            1: (0.75, 0.25), # 75% HMM, 25% N-gram
            2: (0.5, 0.5),   # Equal blend
            3: (0.25, 0.75), # 25% HMM, 75% N-gram
            4: (0.0, 1.0),   # 100% N-gram
        }
        
        self.stats = {
            'episode_rewards': [],
            'episode_wins': [],
            'action_distribution': {i: 0 for i in range(num_actions)},
        }
    
    def encode_state(self, hmm_probs, ngram_probs, game_state, guessed_letters):
        """
        Encode the current state for the meta-learner.
        
        Features:
        1. Progress (masked_word length / total length)
        2. Lives remaining (normalized)
        3. HMM entropy
        4. N-gram entropy  
        5. Model agreement (cosine similarity)
        6. HMM top confidence
        7. N-gram top confidence
        8. Number of guessed letters (normalized)
        9. Early game indicator (< 3 guesses)
        10. Late game indicator (< 2 lives)
        """
        # Convert to arrays - handle both dict and array formats
        # Ensure we have 26-element arrays for a-z
        if isinstance(hmm_probs, dict):
            # HMM might have uppercase letters, normalize to lowercase
            hmm_dict_lower = {}
            for k, v in hmm_probs.items():
                hmm_dict_lower[k.lower()] = v
            hmm_arr = np.array([hmm_dict_lower.get(chr(ord('a') + i), 0) for i in range(26)])
        else:
            hmm_arr = np.array(hmm_probs)
            
        if isinstance(ngram_probs, dict):
            ngram_arr = np.array([ngram_probs.get(chr(ord('a') + i), 0) for i in range(26)])
        else:
            ngram_arr = np.array(ngram_probs)
        
        # Feature 1: Progress
        masked_word = game_state.get('masked_word', '')
        total_letters = len(masked_word.replace('_', '').replace(' ', ''))
        word_length = len(masked_word.replace(' ', ''))
        progress = total_letters / max(word_length, 1)
        
        # Feature 2: Lives remaining
        lives = game_state.get('lives', 6) / 6.0
        
        # Feature 3-4: Entropy (uncertainty measure)
        hmm_entropy = -np.sum(hmm_arr * np.log(hmm_arr + 1e-10))
        ngram_entropy = -np.sum(ngram_arr * np.log(ngram_arr + 1e-10))
        hmm_entropy = hmm_entropy / np.log(26)  # Normalize
        ngram_entropy = ngram_entropy / np.log(26)
        
        # Feature 5: Model agreement
        hmm_norm = hmm_arr / (np.linalg.norm(hmm_arr) + 1e-10)
        ngram_norm = ngram_arr / (np.linalg.norm(ngram_arr) + 1e-10)
        agreement = np.dot(hmm_norm, ngram_norm)
        
        # Feature 6-7: Top confidence
        hmm_confidence = np.max(hmm_arr)
        ngram_confidence = np.max(ngram_arr)
        
        # Feature 8: Guessed letters
        num_guessed = len(guessed_letters) / 26.0
        
        # Feature 9-10: Game phase indicators
        early_game = 1.0 if len(guessed_letters) < 3 else 0.0
        late_game = 1.0 if game_state.get('lives', 6) < 2 else 0.0
        
        state = np.array([
            progress,
            lives,
            hmm_entropy,
            ngram_entropy,
            agreement,
            hmm_confidence,
            ngram_confidence,
            num_guessed,
            early_game,
            late_game
        ], dtype=np.float32)
        
        return state
    
    def select_action(self, state, eval_mode=False):
        """Select action using epsilon-greedy policy"""
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax(1).item()
        else:
            action = random.randint(0, self.num_actions - 1)
        
        self.stats['action_distribution'][action] += 1
        return action
    
    def get_weights(self, action):
        """Convert action to blending weights"""
        return self.action_to_weights[action]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step using experience replay"""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save the meta-learner"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'stats': self.stats,
        }, filepath)
        print(f"✅ Saved meta-learner to {filepath}")
    
    def load(self, filepath):
        """Load the meta-learner"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.stats = checkpoint['stats']
        print(f"✅ Loaded meta-learner from {filepath}")


class RLEnsembleAgent:
    """
    Hangman agent that uses RL to blend HMM and N-gram predictions.
    """
    
    def __init__(self, hmm_predictor, ngram_model, meta_learner=None):
        self.hmm = hmm_predictor
        self.ngram = ngram_model
        self.meta_learner = meta_learner or RLMetaLearner()
        self.guessed_letters = set()
        
        # Episode tracking
        self.current_state = None
        self.current_action = None
        self.episode_reward = 0
        
    def reset(self):
        """Reset for new game"""
        self.guessed_letters = set()
        self.current_state = None
        self.current_action = None
        self.episode_reward = 0
    
    def get_action(self, game_state, eval_mode=False):
        """
        Get next letter guess using RL-guided ensemble.
        
        Returns: (letter, debug_info)
        """
        masked_word = game_state.get('masked_word', '')
        
        # Track wrong letters from guessed - revealed
        revealed = set(c for c in masked_word if c != '_' and c != ' ')
        wrong_letters = list(self.guessed_letters - revealed)
        guessed_set = set(g.upper() for g in self.guessed_letters)
        
        # Get predictions from both models
        hmm_probs = self.hmm.predict(masked_word, guessed_set, wrong_letters)
        ngram_probs = self.ngram.predict_letter_probs(masked_word, list(self.guessed_letters))
        
        # Encode state for meta-learner
        state = self.meta_learner.encode_state(
            hmm_probs, ngram_probs, game_state, self.guessed_letters
        )
        
        # Select blending action using RL
        action = self.meta_learner.select_action(state, eval_mode=eval_mode)
        hmm_weight, ngram_weight = self.meta_learner.get_weights(action)
        
        # Blend predictions
        all_letters = set('abcdefghijklmnopqrstuvwxyz')
        available = all_letters - self.guessed_letters
        
        best_letter = None
        best_score = -1
        
        for letter in available:
            # HMM dict has uppercase keys, ngram is array indexed by a-z
            hmm_score = hmm_probs.get(letter.upper(), 0) if isinstance(hmm_probs, dict) else hmm_probs[ord(letter) - ord('a')]
            ngram_score = ngram_probs[ord(letter) - ord('a')] if isinstance(ngram_probs, np.ndarray) else ngram_probs.get(letter, 0)
            
            score = hmm_weight * hmm_score + ngram_weight * ngram_score
            if score > best_score:
                best_score = score
                best_letter = letter
        
        # Store state and action for training
        self.current_state = state
        self.current_action = action
        
        # Debug info
        if isinstance(hmm_probs, dict):
            hmm_top = max(hmm_probs, key=hmm_probs.get)
        else:
            hmm_top = chr(ord('a') + np.argmax(hmm_probs))
            
        if isinstance(ngram_probs, dict):
            ngram_top = max(ngram_probs, key=ngram_probs.get)
        else:
            ngram_top = chr(ord('a') + np.argmax(ngram_probs))
        
        debug_info = {
            'hmm_weight': hmm_weight,
            'ngram_weight': ngram_weight,
            'action': action,
            'hmm_top': hmm_top,
            'ngram_top': ngram_top,
        }
        
        return best_letter, debug_info
    
    def update(self, letter, correct, game_state, done):
        """Update after receiving feedback"""
        self.guessed_letters.add(letter)
        
        # Calculate reward
        if done:
            if game_state.get('lives', 0) > 0:  # Win
                reward = 100
            else:  # Loss
                reward = -50
        else:
            reward = 10 if correct else -5
        
        self.episode_reward += reward
        
        # Get next state
        if not done:
            masked_word = game_state.get('masked_word', '')
            revealed = set(c for c in masked_word if c != '_' and c != ' ')
            wrong_letters = list(self.guessed_letters - revealed)
            guessed_set = set(g.upper() for g in self.guessed_letters)
            hmm_probs = self.hmm.predict(masked_word, guessed_set, wrong_letters)
            ngram_probs = self.ngram.predict_letter_probs(masked_word, list(self.guessed_letters))
            next_state = self.meta_learner.encode_state(
                hmm_probs, ngram_probs, game_state, self.guessed_letters
            )
        else:
            next_state = np.zeros(self.meta_learner.state_dim, dtype=np.float32)
        
        # Store transition
        if self.current_state is not None:
            self.meta_learner.store_transition(
                self.current_state, self.current_action, reward, next_state, done
            )
