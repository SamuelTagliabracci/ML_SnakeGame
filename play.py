import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Constants
BLOCK_SIZE = 20
WIDTH = 800  # Wider for 2 players
HEIGHT = 400
HUMAN_SPEED = 10  # Normal speed for human player

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

# Define directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Initialize pygame
pygame.init()

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Snake:
    def __init__(self, is_ai=False):
        self.is_ai = is_ai
        
        # Set starting positions - AI on left, human on right
        if is_ai:
            self.head = [WIDTH // 4, HEIGHT // 2]
            self.color = BLUE
            self.food_color = PURPLE
        else:
            self.head = [3 * WIDTH // 4, HEIGHT // 2]
            self.color = GREEN
            self.food_color = RED
            
        self.snake = [self.head.copy()]
        self.direction = RIGHT if is_ai else LEFT  # Start facing each other
        self.score = 0
        self.food = self._place_food()
        self.game_over = False
        self.prev_distance = self._get_food_distance()
        
    def _place_food(self):
        # For AI, food on left side; for human, food on right side
        if self.is_ai:
            x = random.randint(0, (WIDTH // 2 - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        else:
            x = random.randint(WIDTH // 2 // BLOCK_SIZE, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            
        y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        food_position = [x, y]
        
        # Ensure food isn't placed on either snake
        if food_position in self.snake:
            return self._place_food()
            
        return food_position
    
    def _is_collision(self, pt=None, other_snake=None):
        if pt is None:
            pt = self.head
            
        # Hit boundary
        if pt[0] < 0 or pt[0] >= WIDTH or pt[1] < 0 or pt[1] >= HEIGHT:
            return True


        # Hit center line
        if self.is_ai and pt[0] >= WIDTH // 2:  # AI snake trying to cross to right side
            return True
        elif not self.is_ai and pt[0] < WIDTH // 2:  # Human snake trying to cross to left side
            return True
            
        # Hit other snake (if provided)
        # Hit itself
        if pt in self.snake[1:]:
            return True
        #if other_snake and pt in other_snake.snake:
            #return True
            
        return False
    
    def _get_food_distance(self):
        head_x, head_y = self.head
        food_x, food_y = self.food
        return abs(head_x - food_x) + abs(head_y - food_y)
    
    def _get_state(self, other_snake):
        head_x, head_y = self.head
        food_x, food_y = self.food
        
        # Check for danger in each direction
        point_u = [head_x, head_y - BLOCK_SIZE]
        point_r = [head_x + BLOCK_SIZE, head_y]
        point_d = [head_x, head_y + BLOCK_SIZE]
        point_l = [head_x - BLOCK_SIZE, head_y]
        
        # Current direction as a one-hot encoding
        dir_u = self.direction == UP
        dir_r = self.direction == RIGHT
        dir_d = self.direction == DOWN
        dir_l = self.direction == LEFT
        
        # Create the state vector (11 values)
        state = [
            # Danger straight ahead
            (dir_u and self._is_collision(point_u, other_snake)) or
            (dir_r and self._is_collision(point_r, other_snake)) or
            (dir_d and self._is_collision(point_d, other_snake)) or
            (dir_l and self._is_collision(point_l, other_snake)),
            
            # Danger to the right
            (dir_u and self._is_collision(point_r, other_snake)) or
            (dir_r and self._is_collision(point_d, other_snake)) or
            (dir_d and self._is_collision(point_l, other_snake)) or
            (dir_l and self._is_collision(point_u, other_snake)),
            
            # Danger to the left
            (dir_u and self._is_collision(point_l, other_snake)) or
            (dir_r and self._is_collision(point_u, other_snake)) or
            (dir_d and self._is_collision(point_r, other_snake)) or
            (dir_l and self._is_collision(point_d, other_snake)),
            
            # Current direction
            dir_l, dir_r, dir_u, dir_d,
            
            # Food direction
            food_x < head_x,  # food is to the left
            food_x > head_x,  # food is to the right
            food_y < head_y,  # food is up
            food_y > head_y   # food is down
        ]
        
        return np.array(state, dtype=int)
    
    def move(self, action=None, other_snake=None):
        # If game is already over, don't move
        if self.game_over:
            return
        
        # For AI: action is an integer: 0 (straight), 1 (right), 2 (left)
        # For human: action is the direct direction (UP, RIGHT, DOWN, LEFT)
        
        if self.is_ai and action is not None:
            # AI movement logic
            clockwise = [UP, RIGHT, DOWN, LEFT]
            current_direction_idx = clockwise.index(self.direction)
            
            if action == 0:  # Continue straight
                new_direction = clockwise[current_direction_idx]
            elif action == 1:  # Turn right (clockwise)
                next_idx = (current_direction_idx + 1) % 4
                new_direction = clockwise[next_idx]
            else:  # Turn left (counter-clockwise)
                next_idx = (current_direction_idx - 1) % 4
                new_direction = clockwise[next_idx]
                
            self.direction = new_direction
        elif not self.is_ai and action is not None:
            # Human movement logic - direct direction
            # Prevent 180-degree turns
            if (action == UP and self.direction != DOWN) or \
               (action == DOWN and self.direction != UP) or \
               (action == LEFT and self.direction != RIGHT) or \
               (action == RIGHT and self.direction != LEFT):
                self.direction = action
        
        # Move the head based on direction
        x, y = self.head
        if self.direction == UP:
            y -= BLOCK_SIZE
        elif self.direction == DOWN:
            y += BLOCK_SIZE
        elif self.direction == LEFT:
            x -= BLOCK_SIZE
        elif self.direction == RIGHT:
            x += BLOCK_SIZE
        
        self.head = [x, y]
        
        # Check for collision
        if self._is_collision(other_snake=other_snake):
            self.game_over = True
            return
        
        # Check if snake got food
        if self.head == self.food:
            self.score += 1
            self.snake.append(self.head.copy())
            self.food = self._place_food()
        else:
            # Move the snake: remove the tail and add the new head position
            self.snake.pop(0)
            self.snake.append(self.head.copy())

class AIAgent:
    def __init__(self):
        self.model = QNetwork(11, 256, 3)  # Input size: 11, Hidden size: 256, Output size: 3 actions
        self.load_model()
    
    def load_model(self):
        # Load the best trained model
        model_path = 'models/best_model.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()  # Set to evaluation mode
            print("Loaded AI model successfully")
        else:
            print(f"Warning: Model file {model_path} not found. AI will use random actions.")
    
    def get_action(self, state):
        # Get action from the model
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            prediction = self.model(state_tensor)
            return torch.argmax(prediction).item()

class Game:
    def __init__(self):
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Snake Game - Human vs AI')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 25)
        self.font_large = pygame.font.SysFont('arial', 40)
        self.reset(keep_scores=False)
    
    def reset(self, keep_scores=True):
        old_ai_score = self.ai_snake.score if keep_scores else 0
        old_human_score = self.human_snake.score if keep_scores else 0

        self.ai_snake = Snake(is_ai=True)
        self.human_snake = Snake(is_ai=False)
        self.ai_agent = AIAgent()
        self.game_active = True
        self.winner = None
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset(keep_scores=False)
                # Human controls - only change direction, not move
                elif event.key == pygame.K_UP and self.human_snake.direction != DOWN:
                    self.human_snake.direction = UP
                elif event.key == pygame.K_DOWN and self.human_snake.direction != UP:
                    self.human_snake.direction = DOWN
                elif event.key == pygame.K_LEFT and self.human_snake.direction != RIGHT:
                    self.human_snake.direction = LEFT
                elif event.key == pygame.K_RIGHT and self.human_snake.direction != LEFT:
                    self.human_snake.direction = RIGHT
        return True

    def update(self):
        # Check if both snakes are dead
        if self.ai_snake.game_over and self.human_snake.game_over:
            self.game_active = False
            # Determine winner by score
            if self.ai_snake.score > self.human_snake.score:
                self.winner = "AI"
            elif self.human_snake.score > self.ai_snake.score:
                self.winner = "Human"
            else:
                self.winner = "Tie"
            return
        
        # Update AI snake if it's still alive
        if not self.ai_snake.game_over:
            state = self.ai_snake._get_state(self.human_snake)
            action = self.ai_agent.get_action(state)
            self.ai_snake.move(action, self.human_snake)
        
        # Update human snake if it's still alive
        if not self.human_snake.game_over:
            # Move in current direction (no action needed, it will use the current direction)
            self.human_snake.move(None, self.ai_snake)
    
    def render(self):
        self.display.fill(BLACK)
        
        # Draw a dividing line down the middle
        pygame.draw.line(self.display, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)
        
        # Draw AI snake (grayed out if dead)
        for i, pt in enumerate(self.ai_snake.snake):
            color = GRAY if self.ai_snake.game_over else self.ai_snake.color
            pygame.draw.rect(self.display, color, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE), 1)
        
        # Draw human snake (grayed out if dead)
        for i, pt in enumerate(self.human_snake.snake):
            color = GRAY if self.human_snake.game_over else self.human_snake.color
            pygame.draw.rect(self.display, color, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE), 1)
        
        # Draw foods (only if that snake is still alive)
        if not self.ai_snake.game_over:
            pygame.draw.rect(self.display, self.ai_snake.food_color, 
                         pygame.Rect(self.ai_snake.food[0], self.ai_snake.food[1], BLOCK_SIZE, BLOCK_SIZE))
        
        if not self.human_snake.game_over:
            pygame.draw.rect(self.display, self.human_snake.food_color, 
                         pygame.Rect(self.human_snake.food[0], self.human_snake.food[1], BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw scores
        ai_score_text = self.font.render(f"AI Score: {self.ai_snake.score}", True, WHITE)
        human_score_text = self.font.render(f"Your Score: {self.human_snake.score}", True, WHITE)
        self.display.blit(ai_score_text, [10, 10])
        self.display.blit(human_score_text, [WIDTH - 170, 10])
        
        # Draw game over message if both snakes are dead
        if not self.game_active:
            if self.winner == "Tie":
                winner_text = self.font_large.render("Game Over - It's a Tie!", True, YELLOW)
            else:
                winner_text = self.font_large.render(f"Game Over - {self.winner} Wins!", True, YELLOW)
            
            restart_text = self.font.render("Press 'R' to Restart", True, WHITE)
            
            text_rect = winner_text.get_rect(center=(WIDTH/2, HEIGHT/2 - 20))
            restart_rect = restart_text.get_rect(center=(WIDTH/2, HEIGHT/2 + 20))
            
            self.display.blit(winner_text, text_rect)
            self.display.blit(restart_text, restart_rect)
        
        # Draw "Dead" message for each snake if they're dead (but game is still active)
        if self.ai_snake.game_over and self.game_active:
            dead_text = self.font.render("DEAD", True, RED)
            self.display.blit(dead_text, [10, 40])
            
        if self.human_snake.game_over and self.game_active:
            dead_text = self.font.render("DEAD", True, RED)
            self.display.blit(dead_text, [WIDTH - 80, 40])
        
        pygame.display.flip()
    
    def run(self):
        while True:
            if not self.handle_events():
                break
                
            self.update()
            self.render()
            self.clock.tick(HUMAN_SPEED)

if __name__ == "__main__":
    game = Game()
    game.run()
