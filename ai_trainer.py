import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from gnomes_environment import GnomesEnvironment, PlayerType
from dqn_agent import DQNAgent

class AITrainer:
    """AI Training System for Gnomes at Night"""
    def __init__(self, env, save_dir='models'):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Get state size from environment
        dummy_state = env.get_state_vector(PlayerType.HUMAN)
        state_size = len(dummy_state)
        
        print(f"State vector size: {state_size}")
        
        # Create agents
        self.human_agent = DQNAgent(state_size, lr=0.001, epsilon=1.0)
        self.ai_agent = DQNAgent(state_size, lr=0.001, epsilon=1.0)
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': {'human': [], 'ai': []},
            'episode_treasures': [],
            'episode_wall_passes': [],
            'win_rates': {'human': [], 'ai': []},
            'episode_lengths': [],
            'losses': {'human': [], 'ai': []}
        }
    
    def train(self, episodes=2000, save_freq=500, render_freq=1000):
        """Train both agents"""
        print(f"Starting training for {episodes} episodes...")
        print(f"Saving models every {save_freq} episodes")
        
        for episode in range(episodes):
            obs = self.env.reset()
            total_rewards = {'human': 0, 'ai': 0}
            episode_length = 0
            losses = {'human': [], 'ai': []}
            
            # Store previous states for learning
            prev_states = {'human': None, 'ai': None}
            prev_actions = {'human': None, 'ai': None}
            
            while not self.env.state.game_over:
                current_player = self.env.state.current_player
                
                # Get current state
                current_state = self.env.get_state_vector(current_player)
                
                # Choose action
                if current_player == PlayerType.HUMAN:
                    action = self.human_agent.act(current_state)
                    agent = self.human_agent
                    player_key = 'human'
                else:
                    action = self.ai_agent.act(current_state)
                    agent = self.ai_agent
                    player_key = 'ai'
                
                # Take step
                next_obs, reward, done, info = self.env.step(action)
                next_state = self.env.get_state_vector(self.env.state.current_player) if not done else np.zeros_like(current_state)
                episode_length += 1
                
                # Store reward for the acting player
                total_rewards[player_key] += reward
                
                # Learn from previous experience
                if prev_states[player_key] is not None:
                    agent.remember(
                        prev_states[player_key],
                        prev_actions[player_key],
                        reward,
                        current_state,
                        done
                    )
                    loss = agent.replay()
                    if loss > 0:
                        losses[player_key].append(loss)
                
                # Update previous state
                prev_states[player_key] = current_state
                prev_actions[player_key] = action
                
                # Render occasionally
                if episode % render_freq == 0 and self.env.render_mode == 'human':
                    self.env.render()
                    
                if done:
                    break
            
            # Store statistics
            self.training_stats['episode_rewards']['human'].append(total_rewards['human'])
            self.training_stats['episode_rewards']['ai'].append(total_rewards['ai'])
            self.training_stats['episode_treasures'].append(self.env.state.collected_treasures)
            self.training_stats['episode_wall_passes'].append(self.env.state.successful_wall_passes)
            self.training_stats['episode_lengths'].append(episode_length)
            
            # Store losses
            avg_human_loss = np.mean(losses['human']) if losses['human'] else 0
            avg_ai_loss = np.mean(losses['ai']) if losses['ai'] else 0
            self.training_stats['losses']['human'].append(avg_human_loss)
            self.training_stats['losses']['ai'].append(avg_ai_loss)
            
            # Calculate win rates (last 100 episodes)
            if episode >= 99:
                recent_human_rewards = self.training_stats['episode_rewards']['human'][-100:]
                recent_ai_rewards = self.training_stats['episode_rewards']['ai'][-100:]
                
                human_wins = sum(1 for h, a in zip(recent_human_rewards, recent_ai_rewards) if h > a)
                ai_wins = sum(1 for h, a in zip(recent_human_rewards, recent_ai_rewards) if a > h)
                
                self.training_stats['win_rates']['human'].append(human_wins / 100)
                self.training_stats['win_rates']['ai'].append(ai_wins / 100)
            
            # Print progress
            if episode % 100 == 0:
                avg_treasures = np.mean(self.training_stats['episode_treasures'][-100:]) if len(self.training_stats['episode_treasures']) >= 100 else np.mean(self.training_stats['episode_treasures'])
                avg_human_reward = np.mean(self.training_stats['episode_rewards']['human'][-100:]) if len(self.training_stats['episode_rewards']['human']) >= 100 else np.mean(self.training_stats['episode_rewards']['human'])
                avg_ai_reward = np.mean(self.training_stats['episode_rewards']['ai'][-100:]) if len(self.training_stats['episode_rewards']['ai']) >= 100 else np.mean(self.training_stats['episode_rewards']['ai'])
                
                print(f"\nEpisode {episode}")
                print(f"  Avg Treasures: {avg_treasures:.2f}")
                print(f"  Avg Human Reward: {avg_human_reward:.2f} (ε={self.human_agent.epsilon:.3f})")
                print(f"  Avg AI Reward: {avg_ai_reward:.2f} (ε={self.ai_agent.epsilon:.3f})")
                
                if len(self.training_stats['win_rates']['human']) > 0:
                    print(f"  Human Win Rate: {self.training_stats['win_rates']['human'][-1]:.3f}")
                    print(f"  AI Win Rate: {self.training_stats['win_rates']['ai'][-1]:.3f}")
            
            # Save models
            if episode % save_freq == 0 and episode > 0:
                self.save_models(episode)
        
        print("\nTraining completed!")
        self.save_models(episodes)
        self.plot_training_results()
        
        return self.training_stats
    
    def save_models(self, episode):
        """Save both agents"""
        human_path = os.path.join(self.save_dir, f'human_agent_ep{episode}.pth')
        ai_path = os.path.join(self.save_dir, f'ai_agent_ep{episode}.pth')
        
        self.human_agent.save(human_path)
        self.ai_agent.save(ai_path)
        
        # Save training statistics
        stats_path = os.path.join(self.save_dir, f'training_stats_ep{episode}.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(self.training_stats, f)
        
        print(f"Models and stats saved at episode {episode}")
    
    def load_models(self, episode):
        """Load both agents"""
        human_path = os.path.join(self.save_dir, f'human_agent_ep{episode}.pth')
        ai_path = os.path.join(self.save_dir, f'ai_agent_ep{episode}.pth')
        
        if os.path.exists(human_path) and os.path.exists(ai_path):
            self.human_agent.load(human_path)
            self.ai_agent.load(ai_path)
            print(f"Models loaded from episode {episode}")
            
            # Load training statistics
            stats_path = os.path.join(self.save_dir, f'training_stats_ep{episode}.pkl')
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    self.training_stats = pickle.load(f)
                print("Training statistics loaded")
        else:
            print(f"No models found for episode {episode}")
    
    def plot_training_results(self):
        """Plot training statistics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.training_stats['episode_rewards']['human'], label='Human AI', alpha=0.7, color='green')
        axes[0, 0].plot(self.training_stats['episode_rewards']['ai'], label='Machine AI', alpha=0.7, color='blue')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Treasures collected
        axes[0, 1].plot(self.training_stats['episode_treasures'], color='gold', alpha=0.7)
        axes[0, 1].set_title('Treasures Collected per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Treasures')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Win rates
        if len(self.training_stats['win_rates']['human']) > 0:
            episodes_100 = range(100, 100 + len(self.training_stats['win_rates']['human']))
            axes[0, 2].plot(episodes_100, self.training_stats['win_rates']['human'], label='Human AI Win Rate', color='green')
            axes[0, 2].plot(episodes_100, self.training_stats['win_rates']['ai'], label='Machine AI Win Rate', color='blue')
            axes[0, 2].set_title('Win Rates (Rolling 100 Episodes)')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Win Rate')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[1, 0].plot(self.training_stats['episode_lengths'], color='purple', alpha=0.7)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Wall passes
        axes[1, 1].plot(self.training_stats['episode_wall_passes'], color='orange', alpha=0.7)
        axes[1, 1].set_title('Wall Passes per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Wall Passes')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Losses
        if self.training_stats['losses']['human']:
            axes[1, 2].plot(self.training_stats['losses']['human'], label='Human AI Loss', alpha=0.7, color='green')
            axes[1, 2].plot(self.training_stats['losses']['ai'], label='Machine AI Loss', alpha=0.7, color='blue')
            axes[1, 2].set_title('Training Losses')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_agents(self, num_games=10):
        """Evaluate trained agents"""
        # Set agents to evaluation mode (no exploration)
        original_epsilon_human = self.human_agent.epsilon
        original_epsilon_ai = self.ai_agent.epsilon
        
        self.human_agent.epsilon = 0.0
        self.ai_agent.epsilon = 0.0
        
        results = {
            'human_total_reward': 0,
            'ai_total_reward': 0,
            'human_wins': 0,
            'ai_wins': 0,
            'draws': 0,
            'avg_treasures': 0,
            'avg_wall_passes': 0,
            'avg_episode_length': 0
        }
        
        print(f"\n=== Evaluating Agents ({num_games} games) ===")
        
        for game in range(num_games):
            obs = self.env.reset()
            game_rewards = {'human': 0, 'ai': 0}
            step_count = 0
            
            while not self.env.state.game_over and step_count < self.env.max_steps:
                current_player = self.env.state.current_player
                current_state = self.env.get_state_vector(current_player)
                
                # Get action from appropriate agent
                if current_player == PlayerType.HUMAN:
                    action = self.human_agent.act(current_state)
                else:
                    action = self.ai_agent.act(current_state)
                
                # Take step
                obs, reward, done, info = self.env.step(action)
                player_key = 'human' if current_player == PlayerType.HUMAN else 'ai'
                game_rewards[player_key] += reward
                step_count += 1
                
                if done:
                    break
            
            # Record results
            results['human_total_reward'] += game_rewards['human']
            results['ai_total_reward'] += game_rewards['ai']
            results['avg_treasures'] += self.env.state.collected_treasures
            results['avg_wall_passes'] += self.env.state.successful_wall_passes
            results['avg_episode_length'] += step_count
            
            # Determine winner
            if game_rewards['human'] > game_rewards['ai']:
                results['human_wins'] += 1
            elif game_rewards['ai'] > game_rewards['human']:
                results['ai_wins'] += 1
            else:
                results['draws'] += 1
            
            print(f"Game {game+1}: Human={game_rewards['human']:.1f}, AI={game_rewards['ai']:.1f}, "
                  f"Treasures={self.env.state.collected_treasures}, Steps={step_count}")
        
        # Calculate averages
        results['human_avg_reward'] = results['human_total_reward'] / num_games
        results['ai_avg_reward'] = results['ai_total_reward'] / num_games
        results['avg_treasures'] /= num_games
        results['avg_wall_passes'] /= num_games
        results['avg_episode_length'] /= num_games
        
        # Restore original epsilon values
        self.human_agent.epsilon = original_epsilon_human
        self.ai_agent.epsilon = original_epsilon_ai
        
        print(f"\n=== Evaluation Results ===")
        print(f"Human AI - Wins: {results['human_wins']}, Avg Reward: {results['human_avg_reward']:.2f}")
        print(f"Machine AI - Wins: {results['ai_wins']}, Avg Reward: {results['ai_avg_reward']:.2f}")
        print(f"Draws: {results['draws']}")
        print(f"Average Treasures: {results['avg_treasures']:.2f}")
        print(f"Average Wall Passes: {results['avg_wall_passes']:.2f}")
        print(f"Average Episode Length: {results['avg_episode_length']:.1f}")
        
        return results