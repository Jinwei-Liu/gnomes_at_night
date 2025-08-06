import pygame
import time
from gnomes_environment import GnomesEnvironment, PlayerType
from ai_trainer import AITrainer

class AIBattleVisualizer:
    """Visualize AI vs AI battles"""
    
    def __init__(self, trainer, delay=0.8):
        self.trainer = trainer
        self.delay = delay  # Delay between moves (seconds)
        self.env = trainer.env
        
        # Set agents to evaluation mode
        self.trainer.human_agent.epsilon = 0.0
        self.trainer.ai_agent.epsilon = 0.0
        
    def run_battle(self, num_games=5):
        """Run and visualize AI battles"""
        print(f"=== AI Battle Visualization ({num_games} games) ===")
        print("Human AI vs Machine AI")
        print("Press ESC to skip current game, Q to quit")
        
        game_results = []
        
        for game_num in range(num_games):
            print(f"\n--- Starting Game {game_num + 1} ---")
            result = self._play_single_game(game_num + 1)
            game_results.append(result)
            
            if result['quit']:
                break
                
            # Brief pause between games
            if game_num < num_games - 1:
                print("Next game starting in 3 seconds...")
                time.sleep(3)
        
        # Show final results
        self._show_final_results(game_results)
        
        return game_results
    
    def _play_single_game(self, game_number):
        """Play a single game with visualization"""
        obs = self.env.reset()
        game_rewards = {'human': 0, 'ai': 0}
        step_count = 0
        quit_game = False
        skip_game = False
        
        print(f"Game {game_number} - Starting position: {obs['gnome_position']}")
        
        while not self.env.state.game_over and step_count < self.env.max_steps:
            current_player = self.env.state.current_player
            current_state = self.env.get_state_vector(current_player)
            
            # Get action from appropriate agent
            if current_player == PlayerType.HUMAN:
                action = self.trainer.human_agent.act(current_state)
                agent_name = "Human AI"
            else:
                action = self.trainer.ai_agent.act(current_state)
                agent_name = "Machine AI"
            
            # Show action
            action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            print(f"Step {step_count + 1}: {agent_name} chooses {action_names[action]}")
            
            # Take step
            obs, reward, done, info = self.env.step(action)
            player_key = 'human' if current_player == PlayerType.HUMAN else 'ai'
            game_rewards[player_key] += reward
            step_count += 1
            
            # Show step results
            if info['treasure_collected']:
                print(f"  💎 {agent_name} collected a treasure! (+{reward:.1f})")
            elif info['is_wall_pass']:
                print(f"  🧱 {agent_name} passed through a wall! (+{reward:.1f})")
            elif reward > 0:
                print(f"  ➡️  {agent_name} moved closer to treasure (+{reward:.1f})")
            elif reward < 0:
                print(f"  ⬅️  {agent_name} moved away or hit wall ({reward:.1f})")
            
            # Render
            if self.env.render_mode == 'human':
                self.env.render()
                
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_game = True
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            quit_game = True
                            break
                        elif event.key == pygame.K_ESCAPE:
                            skip_game = True
                            break
                
                if quit_game or skip_game:
                    break
                    
                # Wait between moves
                time.sleep(self.delay)
            
            if done:
                break
        
        # Game results
        winner = "Human AI" if game_rewards['human'] > game_rewards['ai'] else \
                "Machine AI" if game_rewards['ai'] > game_rewards['human'] else "Draw"
        
        result = {
            'game_number': game_number,
            'winner': winner,
            'human_reward': game_rewards['human'],
            'ai_reward': game_rewards['ai'],
            'treasures': self.env.state.collected_treasures,
            'wall_passes': self.env.state.successful_wall_passes,
            'steps': step_count,
            'quit': quit_game,
            'skipped': skip_game
        }
        
        print(f"\n--- Game {game_number} Results ---")
        print(f"Winner: {winner}")
        print(f"Human AI Score: {game_rewards['human']:.1f}")
        print(f"Machine AI Score: {game_rewards['ai']:.1f}")
        print(f"Treasures Collected: {self.env.state.collected_treasures}")
        print(f"Wall Passes: {self.env.state.successful_wall_passes}")
        print(f"Steps Taken: {step_count}")
        
        return result
    
    def _show_final_results(self, game_results):
        """Show final battle results"""
        if not game_results:
            return
            
        print(f"\n{'='*50}")
        print("FINAL BATTLE RESULTS")
        print(f"{'='*50}")
        
        human_wins = sum(1 for r in game_results if r['winner'] == 'Human AI')
        ai_wins = sum(1 for r in game_results if r['winner'] == 'Machine AI')
        draws = sum(1 for r in game_results if r['winner'] == 'Draw')
        
        avg_human_score = sum(r['human_reward'] for r in game_results) / len(game_results)
        avg_ai_score = sum(r['ai_reward'] for r in game_results) / len(game_results)
        avg_treasures = sum(r['treasures'] for r in game_results) / len(game_results)
        avg_wall_passes = sum(r['wall_passes'] for r in game_results) / len(game_results)
        avg_steps = sum(r['steps'] for r in game_results) / len(game_results)
        
        print(f"Games Played: {len(game_results)}")
        print(f"\nWin Statistics:")
        print(f"  Human AI Wins: {human_wins}")
        print(f"  Machine AI Wins: {ai_wins}")
        print(f"  Draws: {draws}")
        
        print(f"\nAverage Scores:")
        print(f"  Human AI: {avg_human_score:.2f}")
        print(f"  Machine AI: {avg_ai_score:.2f}")
        
        print(f"\nAverage Game Stats:")
        print(f"  Treasures Collected: {avg_treasures:.2f}")
        print(f"  Wall Passes: {avg_wall_passes:.2f}")
        print(f"  Episode Length: {avg_steps:.1f} steps")
        
        # Determine overall winner
        if human_wins > ai_wins:
            overall_winner = "Human AI"
        elif ai_wins > human_wins:
            overall_winner = "Machine AI"
        else:
            overall_winner = "Draw"
            
        print(f"\n🏆 OVERALL WINNER: {overall_winner} 🏆")
        print(f"{'='*50}")


def main():
    """Main demo function"""
    print("=== Gnomes at Night - AI Battle Demo ===")
    
    # Create environment with visualization
    env = GnomesEnvironment(maze_size=10, max_steps=200, render_mode='human')
    
    # Create trainer and load models
    trainer = AITrainer(env)
    
    # Try to load the latest models
    model_episodes = [2000, 1500, 1000, 500]  # Try these in order
    loaded = False
    
    for episode in model_episodes:
        try:
            trainer.load_models(episode)
            print(f"Successfully loaded models from episode {episode}")
            loaded = True
            break
        except:
            continue
    
    if not loaded:
        print("No trained models found. Training new models first...")
        print("This may take several minutes...")
        
        # Train new models
        env_train = GnomesEnvironment(maze_size=10, max_steps=200, render_mode=None)
        trainer_train = AITrainer(env_train)
        trainer_train.train(episodes=1000, save_freq=500)
        
        # Load the trained models
        trainer.load_models(1000)
        print("Training completed! Starting battle demo...")
    
    # Create visualizer
    visualizer = AIBattleVisualizer(trainer, delay=1.0)  # 1 second between moves
    
    # Run battles
    print("\nStarting AI battle visualization...")
    print("Watch as Human AI (green) battles Machine AI (blue)!")
    results = visualizer.run_battle(num_games=5)
    
    # Cleanup
    pygame.quit()
    print("Demo completed!")


if __name__ == "__main__":
    main()