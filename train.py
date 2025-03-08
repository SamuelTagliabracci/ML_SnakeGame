import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque
import time

# Constants
BLOCK_SIZE = 20
WIDTH = 400
HEIGHT = 400
SPEED = 1000  # Higher value = faster gameplay for training
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9  # discount rate

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

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

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)

class Snake:
    def __init__(self):
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Snake Game - AI Training')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 25)
        self.current_score = 0  # Track score between resets
        self.reset()
    
    def reset(self):
        self.head = [WIDTH // 2, HEIGHT // 2]
        self.snake = [self.head.copy()]
        self.direction = RIGHT
        self.score = self.current_score  # Keep the score from previous game
        self.food = self._place_food()
        self.frame_iteration = 0
        self.prev_distance = self._get_food_distance()
        return self._get_state()
    
    def _place_food(self):
        x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        food_position = [x, y]
        if food_position in self.snake:
            return self._place_food()
        return food_position
    
    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hit boundary
        if pt[0] < 0 or pt[0] >= WIDTH or pt[1] < 0 or pt[1] >= HEIGHT:
            return True
        # Hit itself
        if pt in self.snake[1:]:
            return True
        return False
    
    def _get_food_distance(self):
        head_x, head_y = self.head
        food_x, food_y = self.food
        return abs(head_x - food_x) + abs(head_y - food_y)
    
    def _move(self, action):
        # Action is an integer: 0 (UP), 1 (RIGHT), 2 (DOWN), 3 (LEFT)
        # We don't allow 180-degree turns (e.g., can't go LEFT when currently going RIGHT)
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
    
    def _get_state(self):
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
            (dir_u and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_d)) or
            (dir_l and self._is_collision(point_l)),
            
            # Danger to the right
            (dir_u and self._is_collision(point_r)) or
            (dir_r and self._is_collision(point_d)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)),
            
            # Danger to the left
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_d)),
            
            # Current direction
            dir_l, dir_r, dir_u, dir_d,
            
            # Food direction
            food_x < head_x,  # food is to the left
            food_x > head_x,  # food is to the right
            food_y < head_y,  # food is up
            food_y > head_y   # food is down
        ]
        
        return np.array(state, dtype=int)
    
    def step(self, action):
        self.frame_iteration += 1
        
        # Handle pygame events (close window, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Save the current food distance before moving
        prev_distance = self.prev_distance
        
        # Move the snake
        self._move(action)
        
        # Check if game is over (collision)
        reward = 0
        game_over = False
        
        # Check collision
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return self._get_state(), reward, game_over, self.score
        
        # Check if snake got food
        if self.head == self.food:
            self.score += 1
            self.current_score = self.score  # Update current score
            reward = 10
            self.snake.append(self.head.copy())
            self.food = self._place_food()
        else:
            # Move the snake: remove the tail and add the new head position
            self.snake.pop(0)
            self.snake.append(self.head.copy())
        
        # Update the display
        self._update_ui()
        self.clock.tick(SPEED)
        
        # Smaller reward for moving towards food
        if not game_over:
            current_distance = self._get_food_distance()
            
            if current_distance < prev_distance:
                reward = 1  # Small reward for moving towards food
            
            # Update the previous distance for next step
            self.prev_distance = current_distance
        
        return self._get_state(), reward, game_over, self.score
    
    def _update_ui(self):
        self.display.fill(BLACK)
        
        # Draw snake
        for i, pt in enumerate(self.snake):
            color = BLUE if i == len(self.snake) - 1 else GREEN
            pygame.draw.rect(self.display, color, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE), 1)
        
        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, [0, 0])
        pygame.display.flip()


class Agent:
    def __init__(self):
        # Initialize the model, memory, and other parameters
        self.n_games = 0
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = GAMMA  # Discount factor
        self.memory = ReplayBuffer(MAX_MEMORY)
        self.model = QNetwork(11, 256, 3)  # Input size: 11, Hidden size: 256, Output size: 3 actions
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.last_saved_score = 0
        self.record = 0
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
    
    def get_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # Random action
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        prediction = self.model(state_tensor)
        return torch.argmax(prediction).item()
    
    def train_short_memory(self, state, action, reward, next_state, done):
        # Train on a single step
        self._train_step(state, action, reward, next_state, done)
    
    def train_long_memory(self):
        # Train on a batch from memory
        if len(self.memory) < BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        # Get current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        next_q = self.model(next_states).max(1)[0]
        target_q = rewards + (~dones) * self.gamma * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _train_step(self, state, action, reward, next_state, done):
        # Convert to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        done = torch.tensor([done], dtype=torch.bool)
        
        # Get current Q value
        current_q = self.model(state).gather(1, action.unsqueeze(1))
        
        # Get next Q value
        next_q = self.model(next_state).max(1)[0].unsqueeze(1)
        target_q = reward + (~done) * self.gamma * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def remember(self, state, action, reward, next_state, done):
        # Add experience to memory
        self.memory.push(state, action, reward, next_state, done)
    
    def save_model(self, score):
        # Save model at each milestone (score is multiple of 10)
        milestone = (score // 10) * 10
        if milestone > 0 and milestone > self.last_saved_score:
            model_path = f'models/model_{milestone}.pth'
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved at score milestone {milestone}")
            self.last_saved_score = milestone
            
        # Also save the best model so far
        if score > self.record:
            self.record = score
            best_model_path = 'models/best_model.pth'
            torch.save(self.model.state_dict(), best_model_path)
            print(f"New record! Best model saved at score {score}")


def train():
    # Initialize the environment and agent
    env = Snake()
    agent = Agent()
    
    # Training loop
    scores = []
    mean_scores = []
    total_score = 0
    
    while True:
        # Get the initial state
        state = env.reset()
        
        game_over = False
        while not game_over:
            # Get action from agent
            action = agent.get_action(state)
            
            # Take action and get new state
            next_state, reward, game_over, score = env.step(action)
            
            # Remember the experience
            agent.remember(state, action, reward, next_state, game_over)
            
            # Train short memory (single step)
            agent.train_short_memory(state, action, reward, next_state, game_over)
            
            # Update state
            state = next_state
            
            # If game over, train long memory and adjust parameters
            if game_over:
                agent.n_games += 1
                
                # Train long memory (experience replay)
                agent.train_long_memory()
                
                # Decay epsilon for exploration
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
                
                # Save model at score milestones
                agent.save_model(score)
                
                # Reset score for next game
                current_score = score
                env.current_score = 0  # Reset the score
                
                # Track scores for statistics
                scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                mean_scores.append(mean_score)
                
                print(f'Game: {agent.n_games}, Score: {score}, Record: {agent.record}, Epsilon: {agent.epsilon:.2f}')

                # Quick check for any events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                # Break out of the inner loop to start a new game
                break

if __name__ == "__main__":
    train()
