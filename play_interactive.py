"""
Interactive Hangman - You Judge if Agent is Correct
====================================================
Think of a word and tell the agent if its guesses are right or wrong!
"""

import torch
import pickle
from rl_meta_learner import RLEnsembleAgent, RLMetaLearner
from hmm_model import EnhancedHMMPredictor
from ngram_model import NgramModel


class InteractiveJudgeGame:
    """Hangman where user judges if agent's guess is correct"""
    
    def __init__(self, agent):
        self.agent = agent
        self.max_wrong = 6
        
    def play_game(self, word_length):
        """Play one game where user judges correctness"""
        
        # Initialize game state
        word_state = ['_'] * word_length
        guessed_letters = []
        wrong_guesses = 0
        turn = 0
        
        print("\n" + "="*60)
        print("🎮 HANGMAN - YOU JUDGE THE AGENT'S GUESSES")
        print("="*60)
        print(f"\nWord length: {word_length} letters")
        print(f"Max wrong guesses: {self.max_wrong}")
        print("\nThink of your secret word now!")
        print("You'll tell the agent if each guess is correct.\n")
        input("Press Enter when you're ready to start...")
        
        # Game loop
        while wrong_guesses < self.max_wrong:
            turn += 1
            
            # Display current state
            print(f"\n{'='*60}")
            print(f"Turn {turn}")
            print(f"Word: {' '.join(word_state)}")
            print(f"Guessed letters: {', '.join(sorted(guessed_letters)) if guessed_letters else 'None'}")
            print(f"Wrong guesses: {wrong_guesses}/{self.max_wrong}")
            
            # Check if won
            if '_' not in word_state:
                print(f"\n{'='*60}")
                print("🎉 RL AGENT WINS!")
                print(f"The agent guessed the word in {turn - 1} turns!")
                print(f"Wrong guesses: {wrong_guesses}")
                print("="*60)
                return True, wrong_guesses, turn - 1
            
            # Get agent's guess
            print(f"\n🤖 RL Agent is thinking...")
            
            # Create game state
            game_state = {
                'masked_word': ''.join(word_state),
                'lives_remaining': self.max_wrong - wrong_guesses,
                'guessed_letters': guessed_letters
            }
            
            # Update agent's internal state
            self.agent.guessed_letters = set(guessed_letters)
            
            # Get action
            result = self.agent.get_action(game_state, eval_mode=True)
            if isinstance(result, tuple):
                guess = result[0]
            else:
                guess = result
            
            # Check for repeated guess
            if guess in guessed_letters:
                print(f"⚠️  Agent repeated: {guess} (finding alternative...)")
                for letter in 'ETAOINSHRDLCUMWFGYPBVKJXQZ':
                    if letter not in guessed_letters:
                        guess = letter
                        break
            
            print(f"\n🤖 Agent guesses: {guess}")
            guessed_letters.append(guess)
            
            # Ask user if guess is correct
            while True:
                response = input(f"\nIs '{guess}' in your word? (y/n): ").strip().lower()
                if response in ['y', 'n', 'yes', 'no']:
                    break
                print("Please enter 'y' for yes or 'n' for no")
            
            if response in ['y', 'yes']:
                # Correct guess - ask for positions
                print(f"✅ Correct!")
                
                # Ask how many times it appears
                while True:
                    try:
                        count = int(input(f"How many times does '{guess}' appear? "))
                        if count > 0 and count <= word_length:
                            break
                        print(f"Please enter a number between 1 and {word_length}")
                    except ValueError:
                        print("Please enter a valid number")
                
                # Ask for positions
                print(f"Enter the position(s) where '{guess}' appears (1-{word_length}):")
                print("(Enter positions separated by spaces, e.g., '1 3 5')")
                
                while True:
                    try:
                        pos_input = input("Positions: ").strip()
                        positions = [int(p) - 1 for p in pos_input.split()]  # Convert to 0-indexed
                        
                        # Validate positions
                        if len(positions) != count:
                            print(f"You said '{guess}' appears {count} time(s), but gave {len(positions)} position(s)")
                            continue
                        
                        if all(0 <= p < word_length for p in positions):
                            # Update word state
                            for pos in positions:
                                word_state[pos] = guess
                            break
                        else:
                            print(f"Positions must be between 1 and {word_length}")
                    except (ValueError, IndexError):
                        print("Invalid input. Enter positions as numbers separated by spaces.")
                
                print(f"Word updated: {' '.join(word_state)}")
                
            else:
                # Wrong guess
                wrong_guesses += 1
                print(f"❌ Wrong! '{guess}' is not in the word.")
                print(f"Wrong guesses: {wrong_guesses}/{self.max_wrong}")
        
        # Game over - agent lost
        print(f"\n{'='*60}")
        print("😊 YOU WIN! The RL agent couldn't guess your word!")
        print(f"Final state: {' '.join(word_state)}")
        
        # Ask what the word was
        actual_word = input("\nWhat was your word? ").strip().upper()
        print(f"The word was: {actual_word}")
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
    
    if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
        meta_learner.policy_net.load_state_dict(checkpoint['policy_net'])
    else:
        meta_learner.policy_net.load_state_dict(checkpoint)
    
    meta_learner.policy_net.eval()
    print("  ✓ RL agent loaded")
    
    # Create ensemble agent
    agent = RLEnsembleAgent(hmm_predictor, ngram_model, meta_learner)
    print("\n✓ All models loaded successfully!\n")
    
    return agent


def get_word_length():
    """Get word length from user"""
    while True:
        try:
            length = int(input("\nEnter the length of your secret word (3-20): "))
            if 3 <= length <= 20:
                return length
            print("❌ Please enter a number between 3 and 20")
        except ValueError:
            print("❌ Please enter a valid number")


def main():
    """Main game loop"""
    print("\n" + "="*60)
    print("🎮 INTERACTIVE HANGMAN - YOU JUDGE")
    print("="*60)
    print("\nWelcome! In this game:")
    print("  • Think of a secret word")
    print("  • Tell the game the word length")
    print("  • The RL agent tries to guess letters")
    print("  • YOU decide if each guess is correct!")
    print("  • Tell the agent where correct letters appear")
    print("\nThe RL agent uses:")
    print("  ✓ Hidden Markov Model (position-based patterns)")
    print("  ✓ N-gram Language Model (language patterns)")
    print("  ✓ Reinforcement Learning (meta-learning)")
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
    game = InteractiveJudgeGame(agent)
    
    # Game statistics
    total_games = 0
    agent_wins = 0
    
    # Main game loop
    while True:
        # Get word length
        word_length = get_word_length()
        
        # Play game
        won, wrong_guesses, turns = game.play_game(word_length)
        
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
    print("Thanks for playing Interactive Hangman!")
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
