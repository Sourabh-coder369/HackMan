"""
Interactive Hangman Game with RL Meta-Learner
==============================================
Play Hangman against the trained RL agent!
The RL agent will try to guess your word.
"""

import torch
import pickle
from rl_meta_learner import RLEnsembleAgent, RLMetaLearner
from hmm_model import EnhancedHMMPredictor
from ngram_model import NgramModel


class InteractiveHangmanGame:
    """Interactive Hangman game where RL agent guesses your word"""
    
    def __init__(self, agent):
        self.agent = agent
        self.max_wrong = 6
        
    def display_hangman(self, wrong_guesses):
        """Display ASCII hangman"""
        stages = [
            """
               ------
               |    |
               |
               |
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |    |
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |   /
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |   / \\
               |
            --------
            GAME OVER!
            """
        ]
        return stages[min(wrong_guesses, 6)]
    
    def play_game(self, secret_word):
        """Play one game of Hangman"""
        secret_word = secret_word.upper()
        word_length = len(secret_word)
        
        # Initialize game state
        word_state = ['_'] * word_length
        guessed_letters = []
        wrong_guesses = 0
        turn = 0
        
        print("\n" + "="*60)
        print("🎮 HANGMAN GAME - RL AGENT vs YOUR WORD")
        print("="*60)
        print(f"\nWord length: {word_length} letters")
        print(f"Max wrong guesses: {self.max_wrong}")
        print("\nThe RL agent will try to guess your word!")
        print("\nStarting game...\n")
        
        # Game loop
        while wrong_guesses < self.max_wrong:
            turn += 1
            
            # Display current state
            print(f"\n{'='*60}")
            print(f"Turn {turn}")
            print(self.display_hangman(wrong_guesses))
            print(f"\nWord: {' '.join(word_state)}")
            print(f"Guessed letters: {', '.join(sorted(guessed_letters)) if guessed_letters else 'None'}")
            print(f"Wrong guesses: {wrong_guesses}/{self.max_wrong}")
            
            # Check if won
            if '_' not in word_state:
                print(f"\n{'='*60}")
                print("🎉 RL AGENT WINS!")
                print(f"The word was: {secret_word}")
                print(f"Solved in {turn} turns with {wrong_guesses} wrong guesses")
                print("="*60)
                return True, wrong_guesses, turn
            
            # Get agent's guess
            print(f"\n🤖 RL Agent is thinking...")
            
            # Create game state dictionary
            game_state = {
                'masked_word': ''.join(word_state),
                'lives_remaining': self.max_wrong - wrong_guesses,
                'guessed_letters': guessed_letters
            }
            
            # Update agent's internal guessed letters
            self.agent.guessed_letters = set(guessed_letters)
            
            # Get action (returns tuple: letter, debug_info)
            result = self.agent.get_action(game_state, eval_mode=True)
            if isinstance(result, tuple):
                guess = result[0]
            else:
                guess = result
            
            # Check for repeated guess (shouldn't happen but just in case)
            if guess in guessed_letters:
                print(f"⚠️  Agent repeated a guess: {guess}")
                # Find a new letter
                for letter in 'ETAOINSHRDLCUMWFGYPBVKJXQZ':
                    if letter not in guessed_letters:
                        guess = letter
                        break
            
            print(f"🤖 RL Agent guesses: {guess}")
            
            # Ask user if guess is correct
            while True:
                response = input(f"Is '{guess}' in your word? (y/n): ").strip().lower()
                if response in ['y', 'n', 'yes', 'no']:
                    break
                print("Please enter 'y' for yes or 'n' for no")
            
            # Update game state
            guessed_letters.append(guess)
            
            if response in ['y', 'yes']:
                # Correct guess - automatically fill in the positions
                count = 0
                for i, letter in enumerate(secret_word):
                    if letter == guess:
                        word_state[i] = guess
                        count += 1
                print(f"✅ Correct! '{guess}' appears {count} time(s) in the word!")
            else:
                # Wrong guess
                wrong_guesses += 1
                print(f"❌ Wrong! '{guess}' is not in the word.")
        
        # Game over - agent lost
        print(f"\n{'='*60}")
        print("😊 YOU WIN! The RL agent couldn't guess your word!")
        print(f"The word was: {secret_word}")
        print(f"Letters guessed: {' '.join(word_state)}")
        print("="*60)
        return False, wrong_guesses, turn


def load_rl_agent():
    """Load the trained RL meta-learner"""
    print("Loading RL Meta-Learner...")
    
    # Load corpus
    print("  Loading corpus...")
    with open('Data/corpus.txt', 'r') as f:
        corpus_words = [line.strip().upper() for line in f if line.strip()]
    print(f"  ✓ Loaded {len(corpus_words)} words")
    
    # Load HMM
    print("  Loading HMM model...")
    with open('models/hmm_collection.pkl', 'rb') as f:
        hmm_collection = pickle.load(f)
    hmm_predictor = EnhancedHMMPredictor(hmm_collection, corpus_words)
    print("  ✓ HMM loaded")
    
    # Load N-gram
    print("  Loading N-gram model...")
    ngram_model = NgramModel.load('models/ngram_model.pkl')
    print("  ✓ N-gram loaded")
    
    # Load RL agent
    print("  Loading RL agent weights...")
    meta_learner = RLMetaLearner(state_dim=10, num_actions=5)
    checkpoint = torch.load('models/rl_meta_learner_best.pth', map_location='cpu')
    
    # Load the policy network weights from checkpoint
    if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
        meta_learner.policy_net.load_state_dict(checkpoint['policy_net'])
    else:
        # If checkpoint is just the state dict
        meta_learner.policy_net.load_state_dict(checkpoint)
    
    meta_learner.policy_net.eval()
    print("  ✓ RL agent loaded")
    
    # Create ensemble agent
    agent = RLEnsembleAgent(hmm_predictor, ngram_model, meta_learner)
    print("\n✓ All models loaded successfully!\n")
    
    return agent


def get_word_from_user():
    """Get a valid word from the user"""
    while True:
        word = input("\nEnter your secret word (letters only, 3-20 characters): ").strip()
        
        if not word:
            print("❌ Please enter a word!")
            continue
        
        if not word.isalpha():
            print("❌ Please use only letters (no numbers or symbols)!")
            continue
        
        if len(word) < 3:
            print("❌ Word must be at least 3 letters long!")
            continue
        
        if len(word) > 20:
            print("❌ Word must be at most 20 letters long!")
            continue
        
        return word.upper()


def main():
    """Main game loop"""
    print("\n" + "="*60)
    print("🎮 INTERACTIVE HANGMAN - RL META-LEARNER")
    print("="*60)
    print("\nWelcome! In this game:")
    print("  • You think of a word")
    print("  • The trained RL agent tries to guess it")
    print("  • The agent gets 6 wrong guesses before losing")
    print("\nThe RL agent uses:")
    print("  ✓ Hidden Markov Model (position-based patterns)")
    print("  ✓ N-gram Language Model (language patterns)")
    print("  ✓ Reinforcement Learning (meta-learning to blend models)")
    print("\n" + "="*60)
    
    # Load the RL agent
    try:
        agent = load_rl_agent()
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find required file: {e}")
        print("\nMake sure you have:")
        print("  • Data/corpus.txt")
        print("  • models/hmm_collection.pkl")
        print("  • models/ngram_model.pkl")
        print("  • models/rl_meta_learner_best.pth")
        return
    
    # Create game
    game = InteractiveHangmanGame(agent)
    
    # Game statistics
    total_games = 0
    agent_wins = 0
    
    # Main game loop
    while True:
        # Get word from user
        secret_word = get_word_from_user()
        
        # Play game
        won, wrong_guesses, turns = game.play_game(secret_word)
        
        # Update statistics
        total_games += 1
        if won:
            agent_wins += 1
        
        # Show statistics
        print(f"\n📊 Session Statistics:")
        print(f"  Total games: {total_games}")
        print(f"  RL Agent wins: {agent_wins}")
        print(f"  Your wins: {total_games - agent_wins}")
        print(f"  RL Agent win rate: {100*agent_wins/total_games:.1f}%")
        
        # Play again?
        print("\n" + "="*60)
        choice = input("\nPlay again? (y/n): ").strip().lower()
        if choice != 'y':
            break
    
    # Final message
    print("\n" + "="*60)
    print("Thanks for playing Hangman with the RL Meta-Learner!")
    print(f"Final Score - RL Agent: {agent_wins}, You: {total_games - agent_wins}")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Game interrupted. Thanks for playing!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
