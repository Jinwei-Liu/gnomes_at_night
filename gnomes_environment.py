import numpy as np
import pygame
import random
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Optional, Any, Dict

class PlayerType(Enum):
    HUMAN = 0
    AI = 1

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

@dataclass
class GameState:
    """Game state"""
    gnome_position: Tuple[int, int]
    current_player: PlayerType
    treasure_position: Optional[Tuple[int, int]]
    treasure_visible_to: List[PlayerType]
    collected_treasures: int = 0
    moves_made: int = 0
    game_over: bool = False
    successful_wall_passes: int = 0

class GnomesEnvironment:
    """Gnomes at Night Environment - Dual Player Version"""
    
    def __init__(self, maze_size=10, max_steps=200, render_mode=None):
        self.maze_size = maze_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Generate asymmetric maps
        self.human_maze = self._generate_maze('human')
        self.ai_maze = self._generate_maze('ai')
        
        # Initialize pygame if needed
        if render_mode == 'human':
            self._init_pygame()
        
        self.reset()
    
    def _init_pygame(self):
        """Initialize pygame display"""
        pygame.init()
        self.cell_size = 35
        self.maze_display_size = self.maze_size * self.cell_size
        self.margin = 20
        self.panel_height = 120
        
        # Calculate window size
        self.window_width = self.maze_display_size * 2 + self.margin * 3
        self.window_height = self.maze_display_size + self.panel_height + self.margin * 3
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Gnomes at Night - Dual Player")
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 28)
        self.clock = pygame.time.Clock()
    
    def _generate_maze(self, player_type: str) -> np.ndarray:
        """Generate asymmetric maze"""
        maze = np.zeros((self.maze_size, self.maze_size))
        
        # Boundary walls
        maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
        
        if player_type == 'human':
            walls = [(2, 2), (3, 2), (4, 2), (2, 3), (4, 3), (2, 4), (3, 4), (4, 4),
                    (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (1, 6), (2, 6), (7, 7), (8, 7)]
        else:  # AI map
            walls = [(6, 1), (7, 1), (8, 1), (6, 2), (8, 2), (6, 3), (7, 3), (8, 3),
                    (2, 6), (2, 7), (2, 8), (3, 7), (4, 7), (6, 6), (7, 6), (8, 6),
                    (6, 7), (7, 7), (8, 7), (6, 8), (7, 8), (8, 8)]
        
        for x, y in walls:
            if 0 < x < self.maze_size - 1 and 0 < y < self.maze_size - 1:
                maze[y, x] = 1
        
        return maze
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment"""
        # Find common empty spaces as starting points
        valid_starts = [(x, y) for x in range(1, self.maze_size - 1) 
                       for y in range(1, self.maze_size - 1)
                       if self.human_maze[y, x] == 0 and self.ai_maze[y, x] == 0]
        
        start_pos = random.choice(valid_starts) if valid_starts else (1, 1)
        
        self.state = GameState(
            gnome_position=start_pos,
            current_player=random.choice([PlayerType.HUMAN, PlayerType.AI]),
            treasure_position=None,
            treasure_visible_to=[]
        )
        
        self._spawn_treasure()
        return self.get_observation()
    
    def _spawn_treasure(self):
        """Spawn new treasure"""
        possible_positions = []
        
        for x in range(1, self.maze_size - 1):
            for y in range(1, self.maze_size - 1):
                human_can_see = self.human_maze[y, x] == 0
                ai_can_see = self.ai_maze[y, x] == 0
                
                if human_can_see or ai_can_see:
                    possible_positions.append((x, y, human_can_see, ai_can_see))
        
        if possible_positions:
            x, y, human_can_see, ai_can_see = random.choice(possible_positions)
            self.state.treasure_position = (x, y)
            
            visible_to = []
            if human_can_see: visible_to.append(PlayerType.HUMAN)
            if ai_can_see: visible_to.append(PlayerType.AI)
            
            # If both can see it, randomly choose one
            self.state.treasure_visible_to = [random.choice(visible_to)] if len(visible_to) == 2 else visible_to
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step"""
        if self.state.game_over:
            return self.get_observation(), 0.0, True, {"error": "Game already over"}
        
        action_enum = Action(action)
        old_position = self.state.gnome_position
        current_player = self.state.current_player
        
        # Calculate new position
        new_position = self._calculate_new_position(old_position, action_enum)
        
        # Check movement
        move_successful, is_wall_pass = self._attempt_move(new_position, current_player)
        
        # Calculate reward
        reward = self._calculate_reward(move_successful, is_wall_pass, old_position, new_position)
        
        # Update state
        if move_successful:
            self.state.gnome_position = new_position
            if is_wall_pass:
                self.state.successful_wall_passes += 1
        
        self.state.moves_made += 1
        
        # Check treasure collection
        treasure_collected = self._check_treasure_collection()
        if treasure_collected:
            self.state.collected_treasures += 1
            reward += 10.0
            self._spawn_treasure()
        
        # Switch player
        self.state.current_player = PlayerType.AI if current_player == PlayerType.HUMAN else PlayerType.HUMAN
        
        # Check game end
        done = self.state.moves_made >= self.max_steps or self.state.collected_treasures >= 5
        self.state.game_over = done
        
        info = {
            "move_successful": move_successful,
            "is_wall_pass": is_wall_pass,
            "treasure_collected": treasure_collected,
            "current_player": self.state.current_player.name,
            "acting_player": current_player.name
        }
        
        return self.get_observation(), reward, done, info
    
    def _calculate_new_position(self, position: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """Calculate new position"""
        x, y = position
        moves = {Action.UP: (0, -1), Action.DOWN: (0, 1), 
                Action.LEFT: (-1, 0), Action.RIGHT: (1, 0)}
        dx, dy = moves[action]
        return (x + dx, y + dy)
    
    def _attempt_move(self, new_position: Tuple[int, int], player: PlayerType) -> Tuple[bool, bool]:
        """Attempt to move"""
        x, y = new_position
        
        # Boundary check
        if not (0 <= x < self.maze_size and 0 <= y < self.maze_size):
            return False, False
        
        current_maze = self.human_maze if player == PlayerType.HUMAN else self.ai_maze
        other_maze = self.ai_maze if player == PlayerType.HUMAN else self.human_maze
        
        # Wall passing mechanism: can move if it's empty space for current player
        if current_maze[y, x] == 0:
            is_wall_pass = other_maze[y, x] == 1
            return True, is_wall_pass
        
        return False, False
    
    def _check_treasure_collection(self) -> bool:
        """Check treasure collection"""
        if self.state.treasure_position and self.state.gnome_position == self.state.treasure_position:
            self.state.treasure_position = None
            self.state.treasure_visible_to = []
            return True
        return False
    
    def _calculate_reward(self, move_successful: bool, is_wall_pass: bool, 
                         old_pos: Tuple[int, int], new_pos: Tuple[int, int]) -> float:
        """Calculate reward"""
        reward = 0.1 if move_successful else -0.1
        
        if move_successful and is_wall_pass:
            reward += 2.0
        
        # Treasure distance reward
        if self.state.treasure_position:
            old_dist = abs(old_pos[0] - self.state.treasure_position[0]) + abs(old_pos[1] - self.state.treasure_position[1])
            new_dist = abs(new_pos[0] - self.state.treasure_position[0]) + abs(new_pos[1] - self.state.treasure_position[1])
            
            if new_dist < old_dist:
                reward += 0.5
            elif new_dist > old_dist:
                reward -= 0.2
        
        return reward
    
    def get_observation(self) -> Dict[str, Any]:
        """Get observation"""
        current_player = self.state.current_player
        current_maze = self.human_maze if current_player == PlayerType.HUMAN else self.ai_maze
        
        treasure_info = (self.state.treasure_position 
                        if self.state.treasure_position and current_player in self.state.treasure_visible_to 
                        else None)
        
        return {
            'maze': current_maze.copy(),
            'gnome_position': self.state.gnome_position,
            'current_player': current_player.value,
            'treasure_position': treasure_info,
            'collected_treasures': self.state.collected_treasures,
            'moves_made': self.state.moves_made,
            'successful_wall_passes': self.state.successful_wall_passes
        }
    
    def get_state_vector(self, player_type: PlayerType) -> np.ndarray:
        """Get state as vector for neural network"""
        # Get player's maze view
        current_maze = self.human_maze if player_type == PlayerType.HUMAN else self.ai_maze
        
        # Flatten maze
        maze_vector = current_maze.flatten()
        
        # Position encoding
        pos_vector = np.zeros(self.maze_size * self.maze_size)
        gx, gy = self.state.gnome_position
        pos_vector[gy * self.maze_size + gx] = 1.0
        
        # Treasure encoding (only if visible to this player)
        treasure_vector = np.zeros(self.maze_size * self.maze_size)
        if (self.state.treasure_position and player_type in self.state.treasure_visible_to):
            tx, ty = self.state.treasure_position
            treasure_vector[ty * self.maze_size + tx] = 1.0
        
        # Game state info
        game_info = np.array([
            self.state.collected_treasures / 5.0,  # Normalize
            self.state.moves_made / self.max_steps,  # Normalize
            self.state.successful_wall_passes / 10.0,  # Normalize
            1.0 if player_type == self.state.current_player else 0.0  # Is my turn
        ])
        
        # Concatenate all features
        state_vector = np.concatenate([maze_vector, pos_vector, treasure_vector, game_info])
        
        return state_vector.astype(np.float32)
    
    def render(self, mode='human'):
        """Render environment"""
        if mode == 'human' and hasattr(self, 'screen'):
            self._render_pygame()
        elif mode == 'console':
            self._render_console()
    
    def _render_pygame(self):
        """Pygame rendering - dual frame layout"""
        # Color definitions
        colors = {
            'bg': (240, 245, 250),
            'panel_bg': (255, 255, 255),
            'wall': (70, 70, 70),
            'empty': (250, 250, 250),
            'human': (50, 150, 50),
            'ai': (50, 100, 200),
            'treasure': (255, 200, 0),
            'text': (40, 40, 40),
            'border': (180, 180, 180),
            'active_border': (255, 100, 100),
            'human_active': (100, 200, 100),
            'ai_active': (100, 150, 255)
        }
        
        self.screen.fill(colors['bg'])
        
        # Calculate positions
        human_x = self.margin
        ai_x = self.margin * 2 + self.maze_display_size
        maze_y = self.margin
        
        # Draw player frames
        self._draw_player_frame(human_x, maze_y, PlayerType.HUMAN, colors)
        self._draw_player_frame(ai_x, maze_y, PlayerType.AI, colors)
        
        # Draw mazes
        self._draw_maze_in_frame(self.human_maze, human_x, maze_y, PlayerType.HUMAN, colors)
        self._draw_maze_in_frame(self.ai_maze, ai_x, maze_y, PlayerType.AI, colors)
        
        # Draw bottom information panel
        self._draw_bottom_panel(colors)
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def _draw_player_frame(self, x, y, player_type, colors):
        """Draw player frame"""
        # Frame size
        frame_width = self.maze_display_size
        frame_height = self.maze_display_size + 50  # Extra space for title and controls
        
        # Check if current player
        is_active = (self.state.current_player == player_type)
        
        # Choose colors
        if is_active:
            border_color = colors['active_border']
            bg_color = colors['human_active'] if player_type == PlayerType.HUMAN else colors['ai_active']
            bg_color = tuple(min(255, c + 20) for c in bg_color)  # Slightly brighter
        else:
            border_color = colors['border']
            bg_color = colors['panel_bg']
        
        # Draw background frame
        frame_rect = pygame.Rect(x - 5, y - 35, frame_width + 10, frame_height)
        pygame.draw.rect(self.screen, bg_color, frame_rect)
        pygame.draw.rect(self.screen, border_color, frame_rect, 3 if is_active else 2)
        
        # Player title
        player_name = "Human Player" if player_type == PlayerType.HUMAN else "AI Player"
        title_color = colors['human'] if player_type == PlayerType.HUMAN else colors['ai']
        title_surface = self.title_font.render(player_name, True, title_color)
        title_rect = title_surface.get_rect(center=(x + frame_width // 2, y - 15))
        self.screen.blit(title_surface, title_rect)
        
        # Control instructions
        if player_type == PlayerType.HUMAN:
            control_text = "Controls: W A S D"
        else:
            control_text = "Controls: Arrow Keys"
        
        control_surface = self.small_font.render(control_text, True, colors['text'])
        control_rect = control_surface.get_rect(center=(x + frame_width // 2, y + frame_height - 25))
        self.screen.blit(control_surface, control_rect)
        
        # Current turn indicator
        if is_active:
            indicator_text = "● YOUR TURN"
            indicator_surface = self.font.render(indicator_text, True, colors['active_border'])
            indicator_rect = indicator_surface.get_rect(center=(x + frame_width // 2, y + frame_height - 5))
            self.screen.blit(indicator_surface, indicator_rect)
    
    def _draw_maze_in_frame(self, maze, frame_x, frame_y, player_type, colors):
        """Draw maze within frame"""
        maze_x = frame_x
        maze_y = frame_y
        
        # Draw cells
        for y in range(self.maze_size):
            for x in range(self.maze_size):
                rect = pygame.Rect(maze_x + x * self.cell_size, maze_y + y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                
                # Draw cell
                color = colors['wall'] if maze[y, x] == 1 else colors['empty']
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, colors['border'], rect, 1)
        
        # Draw treasure
        if (self.state.treasure_position and player_type in self.state.treasure_visible_to):
            tx, ty = self.state.treasure_position
            treasure_center = (maze_x + tx * self.cell_size + self.cell_size // 2,
                             maze_y + ty * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(self.screen, colors['treasure'], treasure_center, self.cell_size // 3)
            pygame.draw.circle(self.screen, colors['text'], treasure_center, self.cell_size // 3, 2)
        
        # Draw gnome
        gx, gy = self.state.gnome_position
        gnome_center = (maze_x + gx * self.cell_size + self.cell_size // 2,
                       maze_y + gy * self.cell_size + self.cell_size // 2)
        player_color = colors['human'] if player_type == PlayerType.HUMAN else colors['ai']
        pygame.draw.circle(self.screen, player_color, gnome_center, self.cell_size // 3)
        pygame.draw.circle(self.screen, colors['text'], gnome_center, self.cell_size // 3, 2)
    
    def _draw_bottom_panel(self, colors):
        """Draw bottom information panel"""
        panel_y = self.margin * 2 + self.maze_display_size + 50
        panel_rect = pygame.Rect(self.margin, panel_y, 
                                self.window_width - self.margin * 2, self.panel_height - 50)
        pygame.draw.rect(self.screen, colors['panel_bg'], panel_rect)
        pygame.draw.rect(self.screen, colors['border'], panel_rect, 2)
        
        # Game status information
        info_y = panel_y + 10
        
        # First row of information
        game_info = [
            f"Treasures Collected: {self.state.collected_treasures}/5",
            f"Total Moves: {self.state.moves_made}/{self.max_steps}",
            f"Wall Passes: {self.state.successful_wall_passes}"
        ]
        
        for i, text in enumerate(game_info):
            x = self.margin + 20 + i * 200
            surface = self.small_font.render(text, True, colors['text'])
            self.screen.blit(surface, (x, info_y))
        
        # Treasure information
        if self.state.treasure_position:
            visible_players = [p.name for p in self.state.treasure_visible_to]
            treasure_info = f"Treasure at {self.state.treasure_position} (visible to: {', '.join(visible_players)})"
            treasure_surface = self.small_font.render(treasure_info, True, colors['treasure'])
            self.screen.blit(treasure_surface, (self.margin + 20, info_y + 25))
        
        # Quit instructions
        quit_text = "Press Q to quit"
        quit_surface = self.small_font.render(quit_text, True, colors['text'])
        self.screen.blit(quit_surface, (self.window_width - 150, info_y + 25))
    
    def _render_console(self):
        """Console rendering"""
        print(f"\n=== Step {self.state.moves_made} ===")
        print(f"Current Player: {self.state.current_player.name}")
        print(f"Position: {self.state.gnome_position}")
        print(f"Treasures: {self.state.collected_treasures} | Wall Passes: {self.state.successful_wall_passes}")
        
        if self.state.treasure_position:
            visible = [p.name for p in self.state.treasure_visible_to]
            print(f"Treasure at {self.state.treasure_position} (visible to: {visible})")
        
        # Show current player view
        current_maze = self.human_maze if self.state.current_player == PlayerType.HUMAN else self.ai_maze
        print(f"\n{self.state.current_player.name}'s view:")
        for y in range(self.maze_size):
            row = ""
            for x in range(self.maze_size):
                if (x, y) == self.state.gnome_position:
                    row += "G "
                elif (self.state.treasure_position and (x, y) == self.state.treasure_position 
                      and self.state.current_player in self.state.treasure_visible_to):
                    row += "T "
                else:
                    row += "# " if current_maze[y, x] == 1 else ". "
            print(row)
    
    def play_interactive(self):
        """Interactive dual player game"""
        print("=== Gnomes at Night - Dual Player ===")
        print("Human Player: W A S D keys")
        print("AI Player: Arrow keys")
        print("Q to quit")
        
        if self.render_mode == 'human':
            self._play_dual_pygame()
        else:
            self._play_console()
    
    def _play_dual_pygame(self):
        """Pygame dual player interactive mode"""
        running = True
        
        while running and not self.state.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    action = self._get_action_from_key_dual(event.key)
                    
                    if action is not None:
                        obs, reward, done, info = self.step(action)
                        player_name = "Human" if obs['current_player'] == 0 else "AI"  # Note: this is the next player
                        prev_player = "AI" if obs['current_player'] == 0 else "Human"
                        
                        print(f"{prev_player} -> {Action(action).name}, Reward: {reward:.2f}")
                        if info['is_wall_pass']:
                            print(f"🧱 {prev_player} passed through wall!")
                        if info['treasure_collected']:
                            print(f"💎 {prev_player} collected treasure!")
                        print(f"Now {player_name}'s turn")
                        
                    elif event.key == pygame.K_q:
                        running = False
            
            self.render()
        
        if self.state.game_over:
            print(f"\nGame Over! Final Score: {self.state.collected_treasures} treasures")
            print(f"Total wall passes: {self.state.successful_wall_passes}")
            
            # Show game over screen
            self._show_game_over()
        
        pygame.quit()
    
    def _show_game_over(self):
        """Show game over screen"""
        # Create semi-transparent overlay
        overlay = pygame.Surface((self.window_width, self.window_height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        game_over_text = self.title_font.render("GAME OVER", True, (255, 255, 255))
        score_text = self.font.render(f"Final Score: {self.state.collected_treasures} treasures", True, (255, 255, 255))
        
        # Center display
        game_over_rect = game_over_text.get_rect(center=(self.window_width // 2, self.window_height // 2 - 30))
        score_rect = score_text.get_rect(center=(self.window_width // 2, self.window_height // 2 + 10))
        
        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(score_text, score_rect)
        
        pygame.display.flip()
        pygame.time.wait(3000)  # Wait 3 seconds
    
    def _play_console(self):
        """Console interactive mode"""
        while not self.state.game_over:
            self.render('console')
            
            # Show different control prompts based on current player
            if self.state.current_player == PlayerType.HUMAN:
                controls = "Human controls: 0=Up, 1=Down, 2=Left, 3=Right, q=Quit"
            else:
                controls = "AI controls: 0=Up, 1=Down, 2=Left, 3=Right, q=Quit"
            
            print(f"\n{controls}")
            action_input = input(f"{self.state.current_player.name}'s turn > ").strip().lower()
            
            if action_input == 'q':
                break
            
            try:
                action = int(action_input)
                if 0 <= action <= 3:
                    obs, reward, done, info = self.step(action)
                    print(f"Reward: {reward:.2f}")
                    if info['is_wall_pass']: print("🧱 Wall pass!")
                    if info['treasure_collected']: print("💎 Treasure!")
                else:
                    print("Invalid action!")
            except ValueError:
                print("Please enter 0-3 or 'q'")
        
        print(f"\nFinal Score: {self.state.collected_treasures} treasures")
    
    def _get_action_from_key_dual(self, key):
        """Get action from key - dual player version"""
        # Human player uses WASD
        human_keys = {
            pygame.K_w: 0,      # Up
            pygame.K_s: 1,      # Down  
            pygame.K_a: 2,      # Left
            pygame.K_d: 3       # Right
        }
        
        # AI player uses arrow keys
        ai_keys = {
            pygame.K_UP: 0,     # Up
            pygame.K_DOWN: 1,   # Down
            pygame.K_LEFT: 2,   # Left
            pygame.K_RIGHT: 3   # Right
        }
        
        # Only current player's keys are valid
        if self.state.current_player == PlayerType.HUMAN:
            return human_keys.get(key)
        else:
            return ai_keys.get(key)

def demo():
    """Dual player game demo"""
    print("=== Gnomes at Night - Dual Player Demo ===")
    
    # Create environment
    env = GnomesEnvironment(maze_size=10, render_mode='human')
    
    # Test API
    obs = env.reset()
    current_player_name = "Human" if obs['current_player'] == 0 else "AI"
    print(f"Game started! First player: {current_player_name}")
    print(f"Starting position: {obs['gnome_position']}")
    
    # Interactive game
    print("\nStarting dual player game...")
    print("Human Player uses: W A S D")
    print("AI Player uses: Arrow Keys")
    env.play_interactive()

if __name__ == "__main__":
    demo()