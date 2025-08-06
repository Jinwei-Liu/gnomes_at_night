from gnomes_environment import GnomesEnvironment
from ai_trainer import AITrainer
from demo_ai_battle import AIBattleVisualizer
import pygame

def train_agents():
    """Train the AI agents"""
    print("=== Training AI Agents ===")
    
    # Create environment for training (no visualization to speed up)
    env = GnomesEnvironment(maze_size=10, max_steps=200, render_mode=None)
    
    # Create trainer
    trainer = AITrainer(env)
    
    # Train agents
    print("Starting training... This may take several minutes.")
    stats = trainer.train(episodes=2000, save_freq=500, render_freq=2000)
    
    print("Training completed!")
    return trainer

def demo_trained_agents():
    """Demo the trained agents"""
    print("\n=== Demo Trained Agents ===")
    
    # Create environment with visualization
    env = GnomesEnvironment(maze_size=10, max_steps=200, render_mode='human')
    
    # Create trainer and load models
    trainer = AITrainer(env)
    
    try:
        trainer.load_models(2000)
        print("Loaded trained models successfully!")
    except:
        print("Could not load models. Please train first.")
        return
    
    # Evaluate agents
    print("\nEvaluating agent performance...")
    eval_results = trainer.evaluate_agents(num_games=10)
    
    # Visualize battles
    print("\nStarting visual battle demo...")
    visualizer = AIBattleVisualizer(trainer, delay=0.8)
    battle_results = visualizer.run_battle(num_games=3)
    
    pygame.quit()

def main():
    """Main function"""
    print("=== Gnomes at Night - AI Training & Demo System ===")
    print("1. Train new agents")
    print("2. Demo existing agents")
    print("3. Train and then demo")
    
    while True:
        choice = input("\nSelect option (1/2/3) or 'q' to quit: ").strip().lower()
        
        if choice == 'q':
            print("Goodbye!")
            break
        elif choice == '1':
            trainer = train_agents()
            print("\nTraining completed! Models saved.")
            print("You can now use option 2 to demo the trained agents.")
        elif choice == '2':
            demo_trained_agents()
            print("Demo completed!")
        elif choice == '3':
            print("Training agents first, then running demo...")
            trainer = train_agents()
            print("\nNow running demo with trained agents...")
            demo_trained_agents()
            print("Training and demo completed!")
        else:
            print("Invalid option. Please select 1, 2, 3, or 'q'.")

if __name__ == "__main__":
    main()