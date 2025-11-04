"""
Hangman Game Environment for Reinforcement Learning
"""
import numpy as np
import random
from utils import mask_word, get_letter_set


class HangmanEnvironment:
    """
    Hangman game environment following OpenAI Gym-like interface
    """
    
    def __init__(self, word_list, max_lives=6):
        """
        Args:
            word_list: List of words to use for games
            max_lives: Maximum wrong guesses allowed (default: 6)
        """
        self.word_list = word_list
        self.max_lives = max_lives
        self.alphabet = get_letter_set()
        
        # Current game state
        self.current_word = None
        self.masked_word = None
        self.guessed_letters = set()
        self.wrong_letters = set()
        self.correct_letters = set()
        self.lives_remaining = max_lives
        self.game_over = False
        self.won = False
        self.repeated_guesses = 0
        self.wrong_guesses = 0
        self.num_guesses = 0
        
    def reset(self, word=None):
        """
        Reset environment for new game
        Args:
            word: Specific word to use (if None, random from list)
        Returns:
            Initial state
        """
        if word is None:
            self.current_word = random.choice(self.word_list).upper()
        else:
            self.current_word = word.upper()
        
        self.masked_word = '_' * len(self.current_word)
        self.guessed_letters = set()
        self.wrong_letters = set()
        self.correct_letters = set()
        self.lives_remaining = self.max_lives
        self.game_over = False
        self.won = False
        self.repeated_guesses = 0
        self.wrong_guesses = 0
        self.num_guesses = 0
        
        return self.get_state()
    
    def step(self, action):
        """
        Take action (guess letter)
        Args:
            action: Letter to guess (string)
        Returns:
            state, reward, done, info
        """
        if self.game_over:
            return self.get_state(), 0, True, self.get_info()
        
        letter = action.upper()
        self.num_guesses += 1
        
        # Check if already guessed (repeated guess)
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            reward = -25  # Heavy penalty for repeated guess
            return self.get_state(), reward, self.game_over, self.get_info()
        
        # Add to guessed letters
        self.guessed_letters.add(letter)
        
        # Check if letter is in word
        if letter in self.current_word:
            # Correct guess
            self.correct_letters.add(letter)
            num_revealed = self.current_word.count(letter)
            
            # Update masked word
            self.masked_word = mask_word(self.current_word, self.guessed_letters)
            
            # Calculate reward
            reward = 20 * num_revealed + 5  # More reveals = better reward
            
            # Check if word is complete
            if '_' not in self.masked_word:
                self.game_over = True
                self.won = True
                # Bonus reward for winning with lives remaining
                reward = 100 + (self.lives_remaining * 10)
        
        else:
            # Wrong guess
            self.wrong_letters.add(letter)
            self.wrong_guesses += 1
            self.lives_remaining -= 1
            reward = -15  # Penalty for wrong guess
            
            # Check if game lost
            if self.lives_remaining <= 0:
                self.game_over = True
                self.won = False
                reward = -100  # Heavy penalty for losing
        
        return self.get_state(), reward, self.game_over, self.get_info()
    
    def get_state(self):
        """
        Get current state representation
        Returns dict with all relevant state information
        """
        return {
            'masked_word': self.masked_word,
            'guessed_letters': self.guessed_letters.copy(),
            'wrong_letters': self.wrong_letters.copy(),
            'correct_letters': self.correct_letters.copy(),
            'lives_remaining': self.lives_remaining,
            'word_length': len(self.current_word),
            'num_revealed': len([c for c in self.masked_word if c != '_']),
            'num_unknown': self.masked_word.count('_'),
            'progress': len([c for c in self.masked_word if c != '_']) / len(self.current_word)
        }
    
    def get_info(self):
        """Get additional game information"""
        return {
            'word': self.current_word,
            'won': self.won,
            'wrong_guesses': self.wrong_guesses,
            'repeated_guesses': self.repeated_guesses,
            'num_guesses': self.num_guesses,
            'lives_remaining': self.lives_remaining
        }
    
    def get_legal_actions(self):
        """Get list of legal actions (unguessed letters)"""
        return list(self.alphabet - self.guessed_letters)
    
    def render(self):
        """Print current game state"""
        print(f"\nWord: {self.masked_word}")
        print(f"Guessed: {sorted(self.guessed_letters)}")
        print(f"Wrong: {sorted(self.wrong_letters)}")
        print(f"Lives: {'❤️ ' * self.lives_remaining}{'💀 ' * (self.max_lives - self.lives_remaining)}")
        print(f"Progress: {self.get_state()['progress']:.1%}")


class BatchHangmanEnvironment:
    """
    Batch environment for running multiple games efficiently
    """
    
    def __init__(self, word_list, max_lives=6):
        self.word_list = word_list
        self.max_lives = max_lives
    
    def evaluate(self, agent, n_games=1000, verbose=False):
        """
        Evaluate agent over multiple games
        Args:
            agent: Agent to evaluate
            n_games: Number of games to play
            verbose: Print progress
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'games_won': 0,
            'games_lost': 0,
            'total_wrong_guesses': 0,
            'total_repeated_guesses': 0,
            'total_guesses': 0,
            'game_details': []
        }
        
        env = HangmanEnvironment(self.word_list, self.max_lives)
        
        for game_num in range(n_games):
            if verbose and (game_num + 1) % 100 == 0:
                print(f"  Playing game {game_num + 1}/{n_games}...")
            
            state = env.reset()
            done = False
            
            while not done:
                action = agent.predict(state, training=False)
                state, reward, done, info = env.step(action)
            
            # Record results
            if info['won']:
                results['games_won'] += 1
            else:
                results['games_lost'] += 1
            
            results['total_wrong_guesses'] += info['wrong_guesses']
            results['total_repeated_guesses'] += info['repeated_guesses']
            results['total_guesses'] += info['num_guesses']
            
            results['game_details'].append({
                'word': info['word'],
                'won': info['won'],
                'wrong_guesses': info['wrong_guesses'],
                'repeated_guesses': info['repeated_guesses'],
                'total_guesses': info['num_guesses']
            })
        
        # Calculate summary statistics
        results['win_rate'] = results['games_won'] / n_games
        results['avg_wrong_guesses'] = results['total_wrong_guesses'] / n_games
        results['avg_repeated_guesses'] = results['total_repeated_guesses'] / n_games
        results['avg_guesses'] = results['total_guesses'] / n_games
        
        # Calculate final score
        results['final_score'] = (
            (results['win_rate'] * n_games) - 
            (results['total_wrong_guesses'] * 5) - 
            (results['total_repeated_guesses'] * 2)
        )
        
        return results
