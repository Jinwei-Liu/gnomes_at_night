"""
Gnomes at Night - Enhanced Maze Design with Cooperative Areas
Featuring enclosed regions that require cross-player cooperation
"""

import numpy as np
import pygame
import gym
from gym import spaces
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Set
import random
import time
from collections import deque

# ============================================================================
# Configuration and Constants
# ============================================================================

class Action(Enum):
    """Action definitions"""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    STAY = 4

@dataclass
class EnvConfig:
    """Environment configuration"""
    # Maze configuration
    maze_level: str = 'A'  # A, B, C, D, or 'custom'
    grid_size: Tuple[int, int] = (8, 8)
    
    # Game parameters
    round_duration: float = 150.0
    num_rounds: int = 3
    num_treasures: int = 12
    
    # Rendering
    render_mode: str = 'human'
    fps: int = 30
    window_size: Tuple[int, int] = (1200, 700)
    
    # RL parameters
    observation_radius: int = 0
    reward_structure: str = 'dense'
    
    # Control mode
    control_mode: str = 'keyboard'

# ============================================================================
# Enhanced Edge-based Maze System
# ============================================================================

class EdgeMaze:
    """Edge-based maze where walls are on edges between cells"""
    
    def __init__(self, size: Tuple[int, int]):
        self.rows, self.cols = size
        # Horizontal walls: (rows+1) x cols
        self.h_walls = np.zeros((self.rows + 1, self.cols), dtype=bool)
        # Vertical walls: rows x (cols+1)
        self.v_walls = np.zeros((self.rows, self.cols + 1), dtype=bool)
        
        # Set outer walls
        self.h_walls[0, :] = True  # Top
        self.h_walls[-1, :] = True  # Bottom
        self.v_walls[:, 0] = True  # Left
        self.v_walls[:, -1] = True  # Right
    
    def add_horizontal_wall(self, row: int, col: int):
        """Add horizontal wall above cell (row, col)"""
        if 0 <= row <= self.rows and 0 <= col < self.cols:
            self.h_walls[row, col] = True
    
    def add_vertical_wall(self, row: int, col: int):
        """Add vertical wall to the left of cell (row, col)"""
        if 0 <= row < self.rows and 0 <= col <= self.cols:
            self.v_walls[row, col] = True
    
    def remove_horizontal_wall(self, row: int, col: int):
        """Remove horizontal wall above cell (row, col)"""
        if 1 <= row < self.rows and 0 <= col < self.cols:  # Don't remove outer walls
            self.h_walls[row, col] = False
    
    def remove_vertical_wall(self, row: int, col: int):
        """Remove vertical wall to the left of cell (row, col)"""
        if 0 <= row < self.rows and 1 <= col < self.cols:  # Don't remove outer walls
            self.v_walls[row, col] = False
    
    def can_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Check if movement is possible between adjacent cells"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # Check bounds
        if not (0 <= to_row < self.rows and 0 <= to_col < self.cols):
            return False
        
        # Check wall between cells
        if from_row == to_row:  # Horizontal movement
            if from_col < to_col:  # Moving right
                return not self.v_walls[from_row, to_col]
            else:  # Moving left
                return not self.v_walls[from_row, from_col]
        elif from_col == to_col:  # Vertical movement
            if from_row < to_row:  # Moving down
                return not self.h_walls[to_row, from_col]
            else:  # Moving up
                return not self.h_walls[from_row, from_col]
        
        return False
    
    def get_accessible_cells(self, start: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Get all cells accessible from start position"""
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            row, col = queue.popleft()
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (row + dr, col + dc)
                if new_pos not in visited and self.can_move((row, col), new_pos):
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return visited

class CooperativeMazeGenerator:
    """Generate mazes with intentional cooperative areas"""

    @staticmethod
    def generate_level_A(size: Tuple[int, int]) -> Tuple[EdgeMaze, EdgeMaze]:
        """Level A: Simple complementary mazes"""
        maze1 = EdgeMaze(size)
        maze2 = EdgeMaze(size)
        rows, cols = size
        
        # Maze 1: Create vertical corridors with some walls
        for col in range(2, cols - 1, 2):
            for row in range(1, rows - 1):
                if row != rows // 2:  # Leave middle open
                    maze1.add_vertical_wall(row, col)
        
        # Add some horizontal connections
        for row in range(2, rows - 1, 2):
            maze1.add_horizontal_wall(row, 1)
            maze1.add_horizontal_wall(row, cols - 2)
        
        # Maze 2: Create horizontal corridors
        for row in range(2, rows - 1, 2):
            for col in range(1, cols - 1):
                if col != cols // 2:  # Leave middle open
                    maze2.add_horizontal_wall(row, col)
        
        # Add some vertical connections
        for col in range(2, cols - 1, 2):
            maze2.add_vertical_wall(1, col)
            maze2.add_vertical_wall(rows - 2, col)
        
        # Create cooperation zones (enclosed in one, open in other)
        if rows >= 4 and cols >= 4:
            # Top-left corner - enclosed in maze1
            maze1.add_vertical_wall(0, 1)
            maze1.add_vertical_wall(1, 1)
            maze1.add_horizontal_wall(2, 0)
            
            # Bottom-right corner - enclosed in maze2
            maze2.add_vertical_wall(rows - 2, cols - 1)
            maze2.add_horizontal_wall(rows - 1, cols - 2)
            maze2.add_vertical_wall(rows - 1, cols - 2)
        
        return maze1, maze2
    
    @staticmethod
    def generate_level_B(size: Tuple[int, int]) -> Tuple[EdgeMaze, EdgeMaze]:
        """Level B: Complementary room patterns"""
        maze1 = EdgeMaze(size)
        maze2 = EdgeMaze(size)
        rows, cols = size
        
        # Create 2x2 rooms in maze1
        for r in range(2, rows, 2):
            for c in range(0, cols):
                maze1.add_horizontal_wall(r, c)
        
        for c in range(2, cols, 2):
            for r in range(0, rows):
                maze1.add_vertical_wall(r, c)
        
        # Create doors in maze1 (ensure connectivity)
        for r in range(2, rows, 2):
            for c in range(1, cols, 2):
                if random.random() < 0.7:  # Most doors open
                    maze1.remove_horizontal_wall(r, c)
        
        for c in range(2, cols, 2):
            for r in range(1, rows, 2):
                if random.random() < 0.7:
                    maze1.remove_vertical_wall(r, c)
        
        # Create offset rooms in maze2
        for r in range(1, rows - 1, 2):
            for c in range(0, cols):
                if r > 0 and r < rows:
                    maze2.add_horizontal_wall(r, c)
        
        for c in range(1, cols - 1, 2):
            for r in range(0, rows):
                if c > 0 and c < cols:
                    maze2.add_vertical_wall(r, c)
        
        # Create different doors in maze2
        for r in range(1, rows, 2):
            for c in range(0, cols, 2):
                if random.random() < 0.7:
                    maze2.remove_horizontal_wall(r, c)
        
        for c in range(1, cols, 2):
            for r in range(0, rows, 2):
                if random.random() < 0.7:
                    maze2.remove_vertical_wall(r, c)
        
        return maze1, maze2
    
    @staticmethod
    def generate_level_C(size: Tuple[int, int]) -> Tuple[EdgeMaze, EdgeMaze]:
        """Level C: Complex interlocking patterns"""
        maze1 = EdgeMaze(size)
        maze2 = EdgeMaze(size)
        rows, cols = size
        
        # Maze 1: Create a branching pattern
        mid_r, mid_c = rows // 2, cols // 2
        
        # Main corridors
        for c in range(cols):
            maze1.add_horizontal_wall(mid_r, c)
        for r in range(rows):
            maze1.add_vertical_wall(r, mid_c)
        
        # Add branches
        for r in [mid_r // 2, mid_r + mid_r // 2]:
            for c in range(1, cols - 1):
                if abs(c - mid_c) > 1:
                    maze1.add_horizontal_wall(r, c)
        
        for c in [mid_c // 2, mid_c + mid_c // 2]:
            for r in range(1, rows - 1):
                if abs(r - mid_r) > 1:
                    maze1.add_vertical_wall(r, c)
        
        # Create openings
        maze1.remove_horizontal_wall(mid_r, mid_c - 1)
        maze1.remove_horizontal_wall(mid_r, mid_c + 1)
        maze1.remove_vertical_wall(mid_r - 1, mid_c)
        maze1.remove_vertical_wall(mid_r + 1, mid_c)
        
        # Open some branch connections
        for r in [mid_r // 2, mid_r + mid_r // 2]:
            maze1.remove_horizontal_wall(r, mid_c)
        for c in [mid_c // 2, mid_c + mid_c // 2]:
            maze1.remove_vertical_wall(mid_r, c)
        
        # Maze 2: Create diagonal-like pattern
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if (r + c) % 3 == 0:
                    if random.random() < 0.5:
                        maze2.add_horizontal_wall(r, c)
                    else:
                        maze2.add_vertical_wall(r, c)
        
        # Add cooperation zones
        if rows >= 6 and cols >= 6:
            # Create pockets that are enclosed in one maze
            corners = [(1, 1), (1, cols - 2), (rows - 2, 1), (rows - 2, cols - 2)]
            for i, (r, c) in enumerate(corners):
                if i % 2 == 0:
                    # Enclose in maze1
                    maze1.add_horizontal_wall(r, c)
                    maze1.add_horizontal_wall(r + 1, c)
                    maze1.add_vertical_wall(r, c)
                    maze1.add_vertical_wall(r, c + 1)
                else:
                    # Enclose in maze2
                    maze2.add_horizontal_wall(r, c)
                    maze2.add_horizontal_wall(r + 1, c)
                    maze2.add_vertical_wall(r, c)
                    maze2.add_vertical_wall(r, c + 1)
        
        return maze1, maze2
    
    @staticmethod
    def generate_level_D(size: Tuple[int, int]) -> Tuple[EdgeMaze, EdgeMaze]:
        """Level D: Maximum complexity with guaranteed cooperation zones"""
        maze1 = EdgeMaze(size)
        maze2 = EdgeMaze(size)
        rows, cols = size
        
        # Create a complex base pattern for maze1
        for r in range(1, rows, 2):
            for c in range(1, cols - 1):
                if random.random() < 0.4:
                    maze1.add_horizontal_wall(r, c)
        
        for c in range(1, cols, 2):
            for r in range(1, rows - 1):
                if random.random() < 0.4:
                    maze1.add_vertical_wall(r, c)
        
        # Different pattern for maze2
        for r in range(2, rows - 1, 2):
            for c in range(1, cols - 1):
                if random.random() < 0.4:
                    maze2.add_horizontal_wall(r, c)
        
        for c in range(2, cols - 1, 2):
            for r in range(1, rows - 1):
                if random.random() < 0.4:
                    maze2.add_vertical_wall(r, c)
        
        # Now add definite cooperation zones (after ensuring connectivity)
        cooperation_zones = []
        
        # Find cells that are accessible in maze2 but can be enclosed in maze1
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                # Check if we can enclose this cell in maze1
                can_enclose = True
                
                # Check if adding walls would disconnect maze1
                temp_accessible_before = len(maze1.get_accessible_cells((0, 0)))
                
                # Try to enclose
                walls_to_add = []
                if not maze1.h_walls[r, c]:
                    walls_to_add.append(('h', r, c))
                if not maze1.h_walls[r + 1, c]:
                    walls_to_add.append(('h', r + 1, c))
                if not maze1.v_walls[r, c]:
                    walls_to_add.append(('v', r, c))
                if not maze1.v_walls[r, c + 1]:
                    walls_to_add.append(('v', r, c + 1))
                
                # Add walls temporarily
                for wall_type, wr, wc in walls_to_add:
                    if wall_type == 'h':
                        maze1.add_horizontal_wall(wr, wc)
                    else:
                        maze1.add_vertical_wall(wr, wc)
                
                # Check if still connected (excluding the enclosed cell)
                test_start = (0, 0) if (r, c) != (0, 0) else (0, 1) if cols > 1 else (1, 0)
                temp_accessible_after = len(maze1.get_accessible_cells(test_start))
                
                # Remove walls
                for wall_type, wr, wc in walls_to_add:
                    if wall_type == 'h':
                        maze1.h_walls[wr, wc] = False
                    else:
                        maze1.v_walls[wr, wc] = False
                
                # If enclosing doesn't break connectivity too much, do it
                if temp_accessible_after >= temp_accessible_before - 5:  # Allow small disconnection
                    cooperation_zones.append((r, c))
                    # Actually add the walls
                    for wall_type, wr, wc in walls_to_add:
                        if wall_type == 'h':
                            maze1.add_horizontal_wall(wr, wc)
                        else:
                            maze1.add_vertical_wall(wr, wc)
                
                if len(cooperation_zones) >= 3:  # Create at least 3 cooperation zones
                    break
            if len(cooperation_zones) >= 3:
                break
        
        return maze1, maze2

    @staticmethod
    def generate_custom_puzzle(size: Tuple[int, int]) -> Tuple[EdgeMaze, EdgeMaze]:
        """Generate a specific puzzle design for testing cooperation"""
        maze1 = EdgeMaze(size)
        maze2 = EdgeMaze(size)
        rows, cols = size
        
        # Create a specific pattern that requires cooperation
        # Example: Create isolated pockets in maze1 that are accessible in maze2
        
        # Create vertical divisions in maze1
        for row in range(rows):
            maze1.add_vertical_wall(row, cols // 3)
            maze1.add_vertical_wall(row, 2 * cols // 3)
        
        # Create horizontal divisions in maze2
        for col in range(cols):
            maze2.add_horizontal_wall(rows // 3, col)
            maze2.add_horizontal_wall(2 * rows // 3, col)
        
        # Create small openings
        maze1.remove_vertical_wall(rows // 2, cols // 3)
        maze1.remove_vertical_wall(rows // 2, 2 * cols // 3)
        
        maze2.remove_horizontal_wall(rows // 3, cols // 2)
        maze2.remove_horizontal_wall(2 * rows // 3, cols // 2)
        
        # Add enclosed chambers
        # Chamber 1: Top-left (enclosed in maze1, open in maze2)
        if rows >= 4 and cols >= 4:
            # Box in maze1
            for r in range(1, 3):
                maze1.add_vertical_wall(r, 2)
            maze1.add_horizontal_wall(1, 1)
            maze1.add_horizontal_wall(3, 1)
            
        # Chamber 2: Bottom-right (enclosed in maze2, open in maze1)
        if rows >= 4 and cols >= 4:
            # Box in maze2
            for c in range(cols - 3, cols - 1):
                maze2.add_horizontal_wall(rows - 2, c)
            maze2.add_vertical_wall(rows - 3, cols - 3)
            maze2.add_vertical_wall(rows - 2, cols - 3)
        
        return maze1, maze2
    
    @staticmethod
    def generate_maze_pair(level: str, size: Tuple[int, int]) -> Tuple[EdgeMaze, EdgeMaze]:
        """Generate maze pair based on difficulty level"""
        if level == 'A':
            return CooperativeMazeGenerator.generate_level_A(size)
        elif level == 'B':
            return CooperativeMazeGenerator.generate_level_B(size)
        elif level == 'C':
            return CooperativeMazeGenerator.generate_level_C(size)
        elif level == 'D':
            return CooperativeMazeGenerator.generate_level_D(size)
        elif level == 'custom':
            return CooperativeMazeGenerator.generate_custom_puzzle(size)
        else:
            # Default to level A
            return CooperativeMazeGenerator.generate_level_A(size)

# ============================================================================
# Game State Management
# ============================================================================

class GameState:
    """Game state management"""
    
    def __init__(self, config: EnvConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset game state"""
        # Generate cooperative maze pair
        self.maze1, self.maze2 = CooperativeMazeGenerator.generate_maze_pair(
            self.config.maze_level, 
            self.config.grid_size
        )
        
        # Initialize gnome position
        self.gnome_pos = (0, 0)
        
        # Game state
        self.current_player = 1
        self.current_round = 1
        self.time_remaining = self.config.round_duration
        self.total_score = 0
        
        # Treasure management - place in cooperation zone
        self.treasure_side = random.choice([1, 2])
        self.treasure_pos = self._place_treasure_in_cooperation_zone()
        self.treasures_collected = 0
        
        # Track visited cells and cooperation
        self.visited_cells = {self.gnome_pos}
        self.cooperation_score = 0
    
    def _place_treasure_in_cooperation_zone(self) -> Tuple[int, int]:
        """Place treasure preferentially in areas that require cooperation"""
        maze = self.maze1 if self.treasure_side == 1 else self.maze2
        other_maze = self.maze2 if self.treasure_side == 1 else self.maze1
        
        # Find cells that are hard to reach on treasure side but easier on other side
        rows, cols = self.config.grid_size
        
        # Get accessibility from gnome position for both mazes
        accessible_treasure_side = maze.get_accessible_cells(self.gnome_pos)
        accessible_other_side = other_maze.get_accessible_cells(self.gnome_pos)
        
        # Find cooperation zones (accessible from other side but not treasure side)
        cooperation_zones = []
        for r in range(rows):
            for c in range(cols):
                pos = (r, c)
                if pos != self.gnome_pos:
                    # Check if this position requires cooperation
                    treasure_side_distance = self._get_path_distance(maze, self.gnome_pos, pos)
                    other_side_distance = self._get_path_distance(other_maze, self.gnome_pos, pos)
                    
                    # Prefer positions that are inaccessible or far on treasure side
                    # but accessible on other side
                    if treasure_side_distance == -1 or treasure_side_distance > 5:
                        if other_side_distance != -1 and other_side_distance < treasure_side_distance:
                            cooperation_zones.append(pos)
        
        # If cooperation zones exist, use them; otherwise random placement
        if cooperation_zones:
            return random.choice(cooperation_zones)
        else:
            # Fallback to any valid position
            valid_positions = [(r, c) for r in range(rows) for c in range(cols) 
                             if (r, c) != self.gnome_pos]
            return random.choice(valid_positions) if valid_positions else (rows - 1, cols - 1)
    
    def _get_path_distance(self, maze: EdgeMaze, start: Tuple[int, int], 
                           goal: Tuple[int, int]) -> int:
        """Get shortest path distance, or -1 if unreachable"""
        if start == goal:
            return 0
        
        visited = {start}
        queue = deque([(start, 0)])
        
        while queue:
            pos, dist = queue.popleft()
            row, col = pos
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (row + dr, col + dc)
                
                if new_pos == goal:
                    return dist + 1
                
                if new_pos not in visited and maze.can_move(pos, new_pos):
                    visited.add(new_pos)
                    queue.append((new_pos, dist + 1))
        
        return -1  # Unreachable
    
    def switch_player(self):
        """Switch active player"""
        self.current_player = 2 if self.current_player == 1 else 1
    
    def collect_treasure(self):
        """Collect treasure and spawn new one"""
        self.treasures_collected += 1
        self.total_score += 1
        
        # Check if this was a cooperation treasure
        maze = self.maze1 if self.treasure_side == 1 else self.maze2
        if self._get_path_distance(maze, (0, 0), self.treasure_pos) == -1:
            self.cooperation_score += 1
        
        # Switch treasure to other side
        self.treasure_side = 2 if self.treasure_side == 1 else 1
        self.treasure_pos = self._place_treasure_in_cooperation_zone()
    
    def get_observation(self, player_id: int) -> np.ndarray:
        """Get observation for specific player"""
        maze = self.maze1 if player_id == 1 else self.maze2
        rows, cols = self.config.grid_size
        
        # Create observation array
        obs = np.zeros((rows + 1, cols + 1, 6), dtype=np.float32)
        
        # Channel 0: Horizontal walls
        obs[:rows+1, :cols, 0] = maze.h_walls
        
        # Channel 1: Vertical walls  
        obs[:rows, :cols+1, 1] = maze.v_walls
        
        # Channel 2: Gnome position
        obs[self.gnome_pos[0], self.gnome_pos[1], 2] = 1
        
        # Channel 3: Treasure (only if on this player's side)
        if self.treasure_side == player_id:
            obs[self.treasure_pos[0], self.treasure_pos[1], 3] = 1
        
        # Channel 4: Reachability map (can gnome reach each cell)
        accessible = maze.get_accessible_cells(self.gnome_pos)
        for pos in accessible:
            obs[pos[0], pos[1], 4] = 1
        
        # Channel 5: Time remaining (normalized)
        obs[:, :, 5] = self.time_remaining / self.config.round_duration
        
        return obs

# ============================================================================
# Main Environment Class (keeping the same as before but with new maze generator)
# ============================================================================

class GnomesAtNightEnv(gym.Env):
    """Gnomes at Night RL Environment with Cooperative Mazes"""
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        
        self.config = config or EnvConfig()
        self.state = GameState(self.config)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(5)
        
        # Observation space (now with 6 channels)
        rows, cols = self.config.grid_size
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(rows + 1, cols + 1, 6),
            dtype=np.float32
        )
        
        # Initialize renderer
        self.renderer = None
        if self.config.render_mode in ['human', 'rgb_array']:
            self.renderer = EnhancedRenderer(self.config)
        
        # Control interface
        self.control_interface = ControlInterface(self.config.control_mode)
        
        # Statistics
        self.episode_stats = {
            'treasures_collected': 0,
            'steps': 0,
            'turns': 0,
            'cooperation_score': 0
        }
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.state.reset()
        self.episode_stats = {
            'treasures_collected': 0,
            'steps': 0,
            'turns': 0,
            'cooperation_score': 0
        }
        
        return self.state.get_observation(self.state.current_player)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action"""
        self.episode_stats['steps'] += 1
        
        # Get current maze for active player
        maze = self.state.maze1 if self.state.current_player == 1 else self.state.maze2
        
        # Compute new position
        old_pos = self.state.gnome_pos
        new_pos = self._compute_new_position(action)
        
        # Check if move is valid
        move_success = False
        if action == Action.STAY.value:
            move_success = True
        elif maze.can_move(old_pos, new_pos):
            self.state.gnome_pos = new_pos
            self.state.visited_cells.add(new_pos)
            move_success = True
        
        # Check treasure collection
        treasure_collected = False
        was_cooperation_treasure = False
        if (self.state.gnome_pos == self.state.treasure_pos and 
            self.state.treasure_side == self.state.current_player):
            # Check if this required cooperation
            if self.state._get_path_distance(maze, (0, 0), self.state.treasure_pos) == -1:
                was_cooperation_treasure = True
                self.episode_stats['cooperation_score'] += 1
            
            self.state.collect_treasure()
            self.episode_stats['treasures_collected'] += 1
            treasure_collected = True
        
        # Calculate reward
        reward = self._calculate_reward(
            old_pos, new_pos, treasure_collected, 
            move_success, was_cooperation_treasure
        )
        
        # Update time
        self.state.time_remaining -= 1.0 / self.config.fps
        
        # Switch turn
        self.state.switch_player()
        self.episode_stats['turns'] += 1
        
        # Check episode end
        done = False
        if self.state.time_remaining <= 0:
            if self.state.current_round >= self.config.num_rounds:
                done = True
            else:
                self.state.current_round += 1
                self.state.time_remaining = self.config.round_duration
        
        # Get observation for next player
        observation = self.state.get_observation(self.state.current_player)
        
        # Info dictionary
        info = {
            'current_player': self.state.current_player,
            'score': self.state.total_score,
            'round': self.state.current_round,
            'time_remaining': self.state.time_remaining,
            'treasure_side': self.state.treasure_side,
            'treasures_collected': self.state.treasures_collected,
            'cooperation_score': self.state.cooperation_score,
            'episode_stats': self.episode_stats.copy()
        }
        
        return observation, reward, done, info
    
    def _compute_new_position(self, action: int) -> Tuple[int, int]:
        """Compute new position based on action"""
        row, col = self.state.gnome_pos
        
        if action == Action.NORTH.value:
            return (row - 1, col)
        elif action == Action.SOUTH.value:
            return (row + 1, col)
        elif action == Action.EAST.value:
            return (row, col + 1)
        elif action == Action.WEST.value:
            return (row, col - 1)
        else:  # STAY
            return (row, col)
    
    def _calculate_reward(self, old_pos: Tuple[int, int], 
                         new_pos: Tuple[int, int],
                         treasure_collected: bool,
                         move_success: bool,
                         was_cooperation: bool) -> float:
        """Calculate reward with cooperation bonus"""
        reward = 0.0
        
        if self.config.reward_structure in ['dense', 'mixed']:
            # Treasure collection reward
            if treasure_collected:
                reward += 10.0
                # Bonus for cooperation treasures
                if was_cooperation:
                    reward += 5.0
            
            # Invalid move penalty
            if not move_success and old_pos == new_pos:
                reward -= 0.5
            
            # Time penalty
            reward -= 0.01
            
            # Exploration bonus
            if new_pos != old_pos and new_pos not in self.state.visited_cells:
                reward += 0.1
            
            # Distance to treasure
            if self.state.treasure_side == self.state.current_player:
                maze = self.state.maze1 if self.state.current_player == 1 else self.state.maze2
                old_dist = self.state._get_path_distance(maze, old_pos, self.state.treasure_pos)
                new_dist = self.state._get_path_distance(maze, new_pos, self.state.treasure_pos)
                
                # Reward getting closer if reachable
                if old_dist != -1 and new_dist != -1:
                    if new_dist < old_dist:
                        reward += 0.05
                # Small reward for being in position to help other player
                elif old_dist == -1:
                    other_maze = self.state.maze2 if self.state.current_player == 1 else self.state.maze1
                    if self.state._get_path_distance(other_maze, new_pos, self.state.treasure_pos) != -1:
                        reward += 0.02
        
        elif self.config.reward_structure == 'sparse':
            if treasure_collected:
                reward += 1.0
                if was_cooperation:
                    reward += 0.5
        
        return reward
    
    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """Render environment"""
        if self.renderer is None:
            return None
        
        mode = mode or self.config.render_mode
        return self.renderer.render(self.state, mode)
    
    def close(self):
        """Close environment"""
        if self.renderer:
            self.renderer.close()

# ============================================================================
# Enhanced Renderer with Better Visuals
# ============================================================================

class EnhancedRenderer:
    """Enhanced renderer with cooperation zone highlighting"""
    
    def __init__(self, config: EnvConfig):
        self.config = config
        self.window = None
        self.clock = None
        
        if config.render_mode in ['human', 'rgb_array']:
            pygame.init()
            pygame.display.set_caption("Gnomes at Night - Cooperative Maze")
            self.window = pygame.display.set_mode(config.window_size)
            self.clock = pygame.time.Clock()
            
            # Enhanced color scheme
            self.colors = {
                'background': (40, 40, 40),
                'grid': (80, 80, 80),
                'wall': (139, 69, 19),
                'gnome': (50, 150, 250),
                'treasure': (255, 215, 0),
                'text': (255, 255, 255),
                'timer_bg': (60, 60, 60),
                'timer_fg': (100, 200, 100),
                'active_bg': (70, 90, 70),
                'inactive_bg': (50, 50, 50),
                'unreachable': (100, 60, 60),  # Highlight unreachable areas
                'reachable': (60, 100, 60),    # Highlight reachable areas
                'cooperation_zone': (150, 100, 200)  # Special cooperation zones
            }
    
    def render(self, state: GameState, mode: str = 'human') -> Optional[np.ndarray]:
        """Render current state with enhanced visuals"""
        if self.window is None:
            return None
        
        # Clear screen
        self.window.fill(self.colors['background'])
        
        # Calculate layout
        width, height = self.config.window_size
        view_width = (width - 60) // 2
        view_height = height - 150
        cell_size = min(view_width // (state.config.grid_size[1] + 1),
                       view_height // (state.config.grid_size[0] + 1))
        
        # Draw both maze views
        self._draw_enhanced_maze_view(state, 1, 20, 100, cell_size)
        self._draw_enhanced_maze_view(state, 2, width // 2 + 20, 100, cell_size)
        
        # Draw UI
        self._draw_ui(state)
        
        # Update display
        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.config.fps)
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)),
                axes=(1, 0, 2)
            )
    
    def _draw_enhanced_maze_view(self, state: GameState, player_id: int,
                                 x_offset: int, y_offset: int, cell_size: int):
        """Draw maze view with reachability highlighting"""
        maze = state.maze1 if player_id == 1 else state.maze2
        rows, cols = state.config.grid_size
        
        # Get reachable cells
        reachable = maze.get_accessible_cells(state.gnome_pos)
        
        # Background for active/inactive player
        is_active = (player_id == state.current_player)
        bg_color = self.colors['active_bg'] if is_active else self.colors['inactive_bg']
        bg_rect = pygame.Rect(x_offset - 10, y_offset - 50,
                             (cols + 1) * cell_size + 20,
                             (rows + 1) * cell_size + 60)
        pygame.draw.rect(self.window, bg_color, bg_rect)
        
        # Title
        font = pygame.font.Font(None, 36)
        title = f"Player {player_id}"
        if is_active:
            title += " [ACTIVE]"
        text = font.render(title, True, self.colors['text'])
        self.window.blit(text, (x_offset, y_offset - 40))
        
        # Draw cells with reachability coloring
        for row in range(rows):
            for col in range(cols):
                x = x_offset + col * cell_size + cell_size // 2
                y = y_offset + row * cell_size + cell_size // 2
                
                # Color based on reachability
                if (row, col) in reachable:
                    cell_color = self.colors['reachable']
                else:
                    cell_color = self.colors['unreachable']
                
                # Draw cell
                cell_rect = pygame.Rect(x, y, cell_size - 1, cell_size - 1)
                pygame.draw.rect(self.window, cell_color, cell_rect)
                pygame.draw.rect(self.window, self.colors['grid'], cell_rect, 1)
        
        # Draw walls (thicker for better visibility)
        wall_thickness = 5
        
        # Horizontal walls
        for row in range(rows + 1):
            for col in range(cols):
                if maze.h_walls[row, col]:
                    x1 = x_offset + col * cell_size + cell_size // 2
                    x2 = x1 + cell_size
                    y = y_offset + row * cell_size + cell_size // 2
                    pygame.draw.line(self.window, self.colors['wall'],
                                   (x1, y), (x2, y), wall_thickness)
        
        # Vertical walls
        for row in range(rows):
            for col in range(cols + 1):
                if maze.v_walls[row, col]:
                    x = x_offset + col * cell_size + cell_size // 2
                    y1 = y_offset + row * cell_size + cell_size // 2
                    y2 = y1 + cell_size
                    pygame.draw.line(self.window, self.colors['wall'],
                                   (x, y1), (x, y2), wall_thickness)
        
        # Draw gnome
        gnome_row, gnome_col = state.gnome_pos
        gnome_x = x_offset + gnome_col * cell_size + cell_size
        gnome_y = y_offset + gnome_row * cell_size + cell_size
        pygame.draw.circle(self.window, self.colors['gnome'],
                         (gnome_x, gnome_y), cell_size // 3)
        pygame.draw.circle(self.window, (255, 255, 255),
                         (gnome_x, gnome_y), cell_size // 3, 2)
        
        # Draw treasure
        if state.treasure_side == player_id:
            treasure_row, treasure_col = state.treasure_pos
            treasure_x = x_offset + treasure_col * cell_size + cell_size
            treasure_y = y_offset + treasure_row * cell_size + cell_size
            
            # Check if treasure is in unreachable zone (cooperation needed)
            if (treasure_row, treasure_col) not in reachable:
                # Draw pulsing effect for cooperation treasures
                pulse = abs(pygame.time.get_ticks() % 1000 - 500) / 500
                radius = int(cell_size // 4 * (1 + pulse * 0.3))
                pygame.draw.circle(self.window, self.colors['cooperation_zone'],
                                 (treasure_x, treasure_y), radius + 5, 3)
            
            # Draw treasure star
            points = []
            for i in range(10):
                angle = i * 36 * 3.14159 / 180
                if i % 2 == 0:
                    r = cell_size // 3
                else:
                    r = cell_size // 6
                px = treasure_x + int(r * np.cos(angle))
                py = treasure_y + int(r * np.sin(angle))
                points.append((px, py))
            pygame.draw.polygon(self.window, self.colors['treasure'], points)
    
    def _draw_ui(self, state: GameState):
        """Draw UI elements"""
        font_large = pygame.font.Font(None, 48)
        font_medium = pygame.font.Font(None, 36)
        font_small = pygame.font.Font(None, 24)
        
        # Title
        title = font_large.render("Gnomes at Night", True, self.colors['text'])
        title_rect = title.get_rect(center=(self.config.window_size[0] // 2, 30))
        self.window.blit(title, title_rect)
        
        # Score and info
        info_y = 60
        info_texts = [
            f"Level: {state.config.maze_level} | Round: {state.current_round}/{state.config.num_rounds}",
            f"Score: {state.total_score} | Cooperation Bonus: {state.cooperation_score}",
            f"Turn: Player {state.current_player} | Treasure on: Player {state.treasure_side}'s side"
        ]
        
        for text in info_texts:
            rendered = font_small.render(text, True, self.colors['text'])
            text_rect = rendered.get_rect(center=(self.config.window_size[0] // 2, info_y))
            self.window.blit(rendered, text_rect)
            info_y += 20
        
        # Time bar
        timer_width = 600
        timer_height = 20
        timer_x = (self.config.window_size[0] - timer_width) // 2
        timer_y = self.config.window_size[1] - 80
        
        pygame.draw.rect(self.window, self.colors['timer_bg'],
                        (timer_x, timer_y, timer_width, timer_height))
        
        time_ratio = max(0, state.time_remaining / state.config.round_duration)
        pygame.draw.rect(self.window, self.colors['timer_fg'],
                        (timer_x, timer_y, timer_width * time_ratio, timer_height))
        
        time_text = font_small.render(f"Time: {state.time_remaining:.1f}s", 
                                     True, self.colors['text'])
        time_rect = time_text.get_rect(center=(self.config.window_size[0] // 2, 
                                              timer_y - 15))
        self.window.blit(time_text, time_rect)
        
        # Controls
        controls = "Controls: Arrow Keys = Move | Space = Stay | Tab = Switch View | ESC = Quit"
        control_text = font_small.render(controls, True, self.colors['text'])
        control_rect = control_text.get_rect(
            center=(self.config.window_size[0] // 2, 
                   self.config.window_size[1] - 30)
        )
        self.window.blit(control_text, control_rect)
    
    def close(self):
        """Close renderer"""
        if self.window:
            pygame.quit()

# ============================================================================
# Control Interface
# ============================================================================

class ControlInterface:
    """Control interface"""
    
    def __init__(self, mode: str = 'keyboard'):
        self.mode = mode
    
    def get_action(self) -> int:
        """Get action from keyboard"""
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]:
            return Action.NORTH.value
        elif keys[pygame.K_DOWN]:
            return Action.SOUTH.value
        elif keys[pygame.K_LEFT]:
            return Action.WEST.value
        elif keys[pygame.K_RIGHT]:
            return Action.EAST.value
        elif keys[pygame.K_SPACE]:
            return Action.STAY.value
        
        return Action.STAY.value

# ============================================================================
# Test Functions
# ============================================================================

def play_game():
    """Play the game with enhanced cooperative mazes"""
    print("Starting Gnomes at Night - Cooperative Maze Game")
    print("Use arrow keys to move, Space to stay")
    print("Work together to reach treasures in enclosed areas!")
    print("\nSelect difficulty level:")
    print("A = Easy | B = Medium | C = Hard | D = Expert | custom = Special puzzle")
    
    level = input("Enter level (A/B/C/D/custom): ").strip().upper()
    if level not in ['A', 'B', 'C', 'D', 'CUSTOM']:
        level = 'A'
    
    config = EnvConfig(
        maze_level=level,
        render_mode='human',
        control_mode='keyboard',
        fps=30,
        grid_size=(8, 8) if level != 'D' else (10, 10)
    )
    
    env = GnomesAtNightEnv(config)
    obs = env.reset()
    done = False
    
    print(f"\nStarting Level {level}!")
    print("Red areas = unreachable, Green areas = reachable")
    print("Purple glow = treasure needs cooperation!\n")
    
    running = True
    while running and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, 
                                  pygame.K_RIGHT, pygame.K_SPACE]:
                    action = env.control_interface.get_action()
                    obs, reward, done, info = env.step(action)
                    
                    if info['episode_stats']['treasures_collected'] > 0:
                        print(f"Score: {info['score']} | "
                              f"Cooperation Bonus: {info['cooperation_score']}")
        
        env.render()
    
    if done:
        print(f"\n{'='*50}")
        print(f"GAME OVER!")
        print(f"Final Score: {env.state.total_score}")
        print(f"Cooperation Score: {env.state.cooperation_score}")
        print(f"Total Treasures: {env.state.treasures_collected}")
        print(f"{'='*50}")
    
    env.close()

if __name__ == "__main__":
    play_game()