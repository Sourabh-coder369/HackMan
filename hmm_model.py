"""
Hidden Markov Model for Hangman
Position-based HMM with separate models for different word lengths
"""
import numpy as np
from collections import defaultdict, Counter
import pickle

class PositionBasedHMM:
    """
    Position-based HMM for word prediction
    Hidden States: Letter positions (0, 1, 2, ..., n-1)
    Emissions: Letters (A-Z)
    """
    
    def __init__(self, word_length):
        self.word_length = word_length
        self.n_states = word_length
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.letter_to_idx = {letter: i for i, letter in enumerate(self.alphabet)}
        self.idx_to_letter = {i: letter for i, letter in enumerate(self.alphabet)}
        
        # Model parameters
        self.start_prob = None  # π: Initial state probabilities
        self.transition_prob = None  # A: State transition probabilities
        self.emission_prob = None  # B: Emission probabilities
        
    def train(self, words):
        """
        Train HMM using Maximum Likelihood Estimation
        Args:
            words: List of words of same length
        """
        n_letters = len(self.alphabet)
        
        # Initialize matrices
        self.start_prob = np.zeros(self.n_states)
        self.transition_prob = np.zeros((self.n_states, self.n_states))
        self.emission_prob = np.zeros((self.n_states, n_letters))
        
        # Count occurrences
        for word in words:
            if len(word) != self.word_length:
                continue
                
            for pos, letter in enumerate(word):
                letter_idx = self.letter_to_idx.get(letter, -1)
                if letter_idx == -1:
                    continue
                
                # Start probability (first position)
                if pos == 0:
                    self.start_prob[pos] += 1
                
                # Emission probability (position -> letter)
                self.emission_prob[pos, letter_idx] += 1
                
                # Transition probability (position -> next position)
                if pos < self.n_states - 1:
                    self.transition_prob[pos, pos + 1] += 1
        
        # Normalize to get probabilities
        self.start_prob = self.start_prob / (self.start_prob.sum() + 1e-10)
        
        for i in range(self.n_states):
            total_transitions = self.transition_prob[i].sum()
            if total_transitions > 0:
                self.transition_prob[i] = self.transition_prob[i] / total_transitions
            
            total_emissions = self.emission_prob[i].sum()
            if total_emissions > 0:
                self.emission_prob[i] = self.emission_prob[i] / total_emissions
        
        # Add smoothing to avoid zero probabilities
        self.emission_prob += 1e-6
        self.emission_prob = self.emission_prob / self.emission_prob.sum(axis=1, keepdims=True)
    
    def predict_letter_probabilities(self, masked_word, guessed_letters):
        """
        Predict probability distribution over letters given current game state
        Args:
            masked_word: String with known letters and '_' for unknowns (e.g., "_PP_E")
            guessed_letters: Set of already guessed letters
        Returns:
            Dictionary {letter: probability}
        """
        if len(masked_word) != self.word_length:
            # Return uniform distribution if length mismatch
            remaining = set(self.alphabet) - guessed_letters
            uniform_prob = 1.0 / len(remaining) if remaining else 0
            return {letter: uniform_prob for letter in remaining}
        
        # Aggregate probabilities from unknown positions
        letter_probs = defaultdict(float)
        unknown_positions = [i for i, char in enumerate(masked_word) if char == '_']
        
        if not unknown_positions:
            return {}
        
        # Sum emission probabilities from all unknown positions
        for pos in unknown_positions:
            for letter_idx, prob in enumerate(self.emission_prob[pos]):
                letter = self.idx_to_letter[letter_idx]
                if letter not in guessed_letters:
                    letter_probs[letter] += prob
        
        # Normalize
        total = sum(letter_probs.values())
        if total > 0:
            letter_probs = {letter: prob / total for letter, prob in letter_probs.items()}
        
        # Ensure all unguessed letters have some probability
        remaining = set(self.alphabet) - guessed_letters
        for letter in remaining:
            if letter not in letter_probs:
                letter_probs[letter] = 1e-6
        
        # Re-normalize
        total = sum(letter_probs.values())
        letter_probs = {letter: prob / total for letter, prob in letter_probs.items()}
        
        return letter_probs


class HMMCollection:
    """
    Collection of HMMs for different word lengths
    """
    
    def __init__(self):
        self.hmms = {}  # {length: HMM}
        self.trained_lengths = set()
    
    def train_all(self, words_by_length, min_words=10):
        """
        Train HMMs for all word lengths
        Args:
            words_by_length: Dict {length: [words]}
            min_words: Minimum number of words required to train HMM
        """
        print("\n🔨 Training HMMs for different word lengths...")
        
        for length, words in sorted(words_by_length.items()):
            if len(words) < min_words:
                print(f"  ⊗ Length {length:2d}: Skipping (only {len(words)} words)")
                continue
            
            hmm = PositionBasedHMM(length)
            hmm.train(words)
            self.hmms[length] = hmm
            self.trained_lengths.add(length)
            print(f"  ✓ Length {length:2d}: Trained on {len(words):5d} words")
        
        print(f"\n✓ Total HMMs trained: {len(self.hmms)}")
    
    def predict(self, masked_word, guessed_letters):
        """
        Get letter probability distribution for current game state
        """
        length = len(masked_word)
        
        if length in self.hmms:
            return self.hmms[length].predict_letter_probabilities(masked_word, guessed_letters)
        else:
            # Fallback to uniform distribution
            remaining = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') - guessed_letters
            uniform_prob = 1.0 / len(remaining) if remaining else 0
            return {letter: uniform_prob for letter in remaining}
    
    def save(self, filepath):
        """Save all HMMs to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ HMMs saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load HMMs from file"""
        with open(filepath, 'rb') as f:
            hmm_collection = pickle.load(f)
        print(f"✓ HMMs loaded from {filepath}")
        return hmm_collection


class EnhancedHMMPredictor:
    """
    Enhanced HMM predictor with word list filtering
    Combines HMM probabilities with actual possible words
    """
    
    def __init__(self, hmm_collection, corpus_words):
        self.hmm_collection = hmm_collection
        self.corpus_words = corpus_words
        self.words_by_length = defaultdict(list)
        
        for word in corpus_words:
            self.words_by_length[len(word)].append(word)
    
    def predict(self, masked_word, guessed_letters, wrong_letters):
        """
        Enhanced prediction using HMM + word filtering
        """
        # Get HMM probabilities
        hmm_probs = self.hmm_collection.predict(masked_word, guessed_letters)
        
        # Filter possible words
        possible_words = self.filter_words(masked_word, wrong_letters)
        
        if not possible_words:
            # Fallback to HMM only
            return hmm_probs
        
        # Count letter frequencies in possible words
        letter_counts = Counter()
        for word in possible_words:
            for letter in set(word):
                if letter not in guessed_letters:
                    letter_counts[letter] += 1
        
        # Normalize
        total = sum(letter_counts.values())
        if total > 0:
            word_probs = {letter: count / total for letter, count in letter_counts.items()}
        else:
            word_probs = {}
        
        # Combine HMM and word-based probabilities
        combined_probs = {}
        all_letters = set(hmm_probs.keys()) | set(word_probs.keys())
        
        alpha = 0.3  # Weight for HMM (tune this!)
        for letter in all_letters:
            hmm_p = hmm_probs.get(letter, 0)
            word_p = word_probs.get(letter, 0)
            combined_probs[letter] = alpha * hmm_p + (1 - alpha) * word_p
        
        # Normalize
        total = sum(combined_probs.values())
        if total > 0:
            combined_probs = {letter: prob / total for letter, prob in combined_probs.items()}
        
        return combined_probs
    
    def filter_words(self, masked_word, wrong_letters):
        """Filter corpus words matching current pattern"""
        length = len(masked_word)
        regex_pattern = masked_word.replace('_', '.')
        
        possible = []
        for word in self.words_by_length[length]:
            if re.match(regex_pattern, word):
                if not any(letter in word for letter in wrong_letters):
                    possible.append(word)
        
        return possible


import re
