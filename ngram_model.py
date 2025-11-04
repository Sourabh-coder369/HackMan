"""
N-gram Language Model for Hangman
Captures local letter dependencies using bigrams and trigrams
Research shows n-grams significantly improve word guessing by understanding common letter patterns
"""

import pickle
from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict, Set, Tuple
import string


class NgramModel:
    """N-gram language model for letter prediction in Hangman"""
    
    def __init__(self, n=3):
        self.n = n  # Maximum n-gram order (1=unigram, 2=bigram, 3=trigram)
        self.unigrams = Counter()  # Letter frequencies
        self.bigrams = defaultdict(Counter)  # P(letter | prev_letter)
        self.trigrams = defaultdict(Counter)  # P(letter | prev_2_letters)
        self.letter_positions = defaultdict(lambda: defaultdict(int))  # Position-aware frequencies
        self.total_letters = 0
        self.vocabulary = set(string.ascii_lowercase)
        
    def train(self, corpus: List[str]):
        """Train n-gram model on word corpus"""
        print(f"Training {self.n}-gram model on {len(corpus)} words...")
        
        for word in corpus:
            word = word.lower().strip()
            if not word or not word.isalpha():
                continue
                
            # Unigram counts
            for i, letter in enumerate(word):
                self.unigrams[letter] += 1
                self.letter_positions[len(word)][letter] += 1
                self.total_letters += 1
                
                # Bigram counts: P(letter | prev_letter)
                if i > 0:
                    prev = word[i-1]
                    self.bigrams[prev][letter] += 1
                    
                # Trigram counts: P(letter | prev_2_letters)
                if i > 1:
                    context = word[i-2:i]
                    self.trigrams[context][letter] += 1
        
        print(f"  Trained on {self.total_letters:,} letters")
        print(f"  Unigrams: {len(self.unigrams)} letters")
        print(f"  Bigrams: {len(self.bigrams)} contexts")
        print(f"  Trigrams: {len(self.trigrams)} contexts")
        
    def get_unigram_probs(self) -> Dict[str, float]:
        """Get unigram probabilities for all letters"""
        total = sum(self.unigrams.values()) or 1
        return {letter: self.unigrams[letter] / total 
                for letter in string.ascii_lowercase}
    
    def get_bigram_probs(self, prev_letter: str) -> Dict[str, float]:
        """Get bigram probabilities P(letter | prev_letter)"""
        if prev_letter not in self.bigrams:
            return self.get_unigram_probs()  # Back-off to unigram
            
        counts = self.bigrams[prev_letter]
        total = sum(counts.values()) or 1
        
        # Smoothing: mix with unigram
        alpha = 0.1  # Smoothing factor
        unigram_probs = self.get_unigram_probs()
        
        probs = {}
        for letter in string.ascii_lowercase:
            bigram_prob = counts[letter] / total
            probs[letter] = (1 - alpha) * bigram_prob + alpha * unigram_probs[letter]
            
        return probs
    
    def get_trigram_probs(self, context: str) -> Dict[str, float]:
        """Get trigram probabilities P(letter | context)"""
        if len(context) < 2 or context not in self.trigrams:
            # Back-off to bigram
            if len(context) >= 1:
                return self.get_bigram_probs(context[-1])
            return self.get_unigram_probs()
            
        counts = self.trigrams[context]
        total = sum(counts.values()) or 1
        
        # Smoothing: mix with bigram
        alpha = 0.1
        bigram_probs = self.get_bigram_probs(context[-1])
        
        probs = {}
        for letter in string.ascii_lowercase:
            trigram_prob = counts[letter] / total
            probs[letter] = (1 - alpha) * trigram_prob + alpha * bigram_probs[letter]
            
        return probs
    
    def predict_letter_probs(self, word_state: str, guessed_letters: Set[str]) -> np.ndarray:
        """
        Predict letter probabilities using n-gram model
        
        Args:
            word_state: Current word state (e.g., "H_LL_")
            guessed_letters: Set of already guessed letters
            
        Returns:
            26-length probability array for a-z
        """
        word_length = len(word_state)
        probs = np.zeros(26)
        
        # Find all contexts in the word
        contexts = []
        for i, char in enumerate(word_state):
            if char == '_':
                # Get context before this position
                context = ""
                if i > 0 and word_state[i-1] != '_':
                    context += word_state[i-1]
                if i > 1 and word_state[i-2] != '_':
                    context = word_state[i-2] + context
                contexts.append((i, context))
        
        # If no gaps, use position-based frequencies
        if not contexts:
            return self._get_position_probs(word_length, guessed_letters)
        
        # Aggregate predictions from all contexts
        for pos, context in contexts:
            if len(context) >= 2:
                context_probs = self.get_trigram_probs(context)
            elif len(context) == 1:
                context_probs = self.get_bigram_probs(context)
            else:
                context_probs = self.get_unigram_probs()
            
            # Add to probabilities
            for letter in string.ascii_lowercase:
                idx = ord(letter) - ord('a')
                probs[idx] += context_probs.get(letter, 0)
        
        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            # Fallback to uniform
            probs = np.ones(26) / 26
        
        # Zero out already guessed letters
        for letter in guessed_letters:
            if letter in string.ascii_lowercase:
                idx = ord(letter) - ord('a')
                probs[idx] = 0
        
        # Re-normalize
        total = probs.sum()
        if total > 0:
            probs /= total
            
        return probs
    
    def _get_position_probs(self, word_length: int, guessed_letters: Set[str]) -> np.ndarray:
        """Get position-based probabilities when no context available"""
        probs = np.zeros(26)
        
        if word_length in self.letter_positions:
            position_counts = self.letter_positions[word_length]
            total = sum(position_counts.values()) or 1
            
            for letter in string.ascii_lowercase:
                if letter not in guessed_letters:
                    idx = ord(letter) - ord('a')
                    probs[idx] = position_counts[letter] / total
        else:
            # Use unigram probabilities
            unigram_probs = self.get_unigram_probs()
            for letter in string.ascii_lowercase:
                if letter not in guessed_letters:
                    idx = ord(letter) - ord('a')
                    probs[idx] = unigram_probs[letter]
        
        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(26) / 26
            
        return probs
    
    def save(self, filepath: str):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n': self.n,
                'unigrams': self.unigrams,
                'bigrams': dict(self.bigrams),
                'trigrams': dict(self.trigrams),
                'letter_positions': dict(self.letter_positions),
                'total_letters': self.total_letters
            }, f)
        print(f"✓ N-gram model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(n=data['n'])
        model.unigrams = data['unigrams']
        model.bigrams = defaultdict(Counter, data['bigrams'])
        model.trigrams = defaultdict(Counter, data['trigrams'])
        model.letter_positions = defaultdict(lambda: defaultdict(int), data['letter_positions'])
        model.total_letters = data['total_letters']
        
        print(f"✓ N-gram model loaded from {filepath}")
        return model


def train_ngram_model():
    """Train and save n-gram model"""
    from utils import CorpusLoader
    
    # Load corpus
    corpus_loader = CorpusLoader('Data/corpus.txt')
    corpus = corpus_loader.load()
    
    # Train trigram model
    model = NgramModel(n=3)
    model.train(corpus)
    
    # Save model
    model.save('models/ngram_model.pkl')
    
    # Test predictions
    print("\n=== Testing N-gram Predictions ===")
    test_cases = [
        ("H_LL_", set(['a', 'b', 'c', 'd'])),  # Should predict 'e' or 'o'
        ("_A_", set(['b', 'c'])),  # Should predict common patterns
        ("TH_", set(['a', 'b', 'c']))  # Should predict 'e' or 'i'
    ]
    
    for word_state, guessed in test_cases:
        probs = model.predict_letter_probs(word_state, guessed)
        top_5 = np.argsort(probs)[-5:][::-1]
        letters = [chr(ord('a') + i) for i in top_5]
        print(f"\nWord: {word_state} | Guessed: {guessed}")
        print(f"Top 5 predictions: {letters}")
        print(f"Probabilities: {[f'{probs[i]:.3f}' for i in top_5]}")


if __name__ == '__main__':
    train_ngram_model()
