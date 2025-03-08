# Snake Game AI with Reinforcement Learning

This project implements a Snake game with reinforcement learning capabilities. It includes two main components:
1. A training mode where an AI learns to play Snake using Q-learning
2. A human vs. AI mode where you can play against the trained model

## Features

- **AI Training Mode**: Train an AI agent using deep Q-learning to master the Snake game
- **Human vs. AI Mode**: Compete against the trained AI in a split-screen battle
- **Reinforcement Learning**: Uses PyTorch to implement a neural network that learns optimal game strategies
- **Model Saving**: Automatically saves models at score milestones and keeps track of the best model

## Gameplay Demo

## 🎥 Demo
| Input                | Output                | Input                | Output                |
|----------------------|-----------------------|----------------------|-----------------------|
|<img src="[examples/image/anime1.png](https://github.com/jixiaozhong/Sonic/blob/main/examples/image/anime1.png?raw=true)" width="360">|<video src="https://github.com/user-attachments/assets/cbad6049-1580-4bc7-af9d-12bfdfbb92e8" width="360" controls> </video>|<img src="[[examples/image/female_diaosu.png](https://github.com/jixiaozhong/Sonic/blob/main/examples/image/anime1.png?raw=true)](https://github.com/jixiaozhong/Sonic/blob/main/examples/image/anime1.png?raw=true)" width="360">|<video src="https://github.com/user-attachments/assets/1923b158-10db-4e2a-a5cb-7b36227ba9db" width="360" controls> </video>|
|<img src="[examples/image/hair.png](https://github.com/jixiaozhong/Sonic/blob/main/examples/image/anime1.png?raw=true)" width="360">|<video src="https://github.com/user-attachments/assets/dcb755c1-de01-4afe-8b4f-0e0b2c2439c1" width="360" controls> </video>|<img src="[examples/image/leonnado.jpg](https://github.com/jixiaozhong/Sonic/blob/main/examples/image/anime1.png?raw=true)" width="360">|<video src="https://github.com/user-attachments/assets/b50e61bb-62d4-469d-b402-b37cda3fbd27" width="360" controls> </video>|


## Requirements

- Python 3.8+
- PyTorch
- Pygame
- NumPy

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/SamuelTagliabracci/ML_SnakeGame.git
   cd ML_SnakeGame
   ```

2. Set up a virtual environment (recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Make sure you have a `models` directory in the project folder (it will be created automatically during training):
   ```
   mkdir -p models
   ```

## Usage

### Training the AI

Run the training script to train the AI agent:

```
python3 train.py
```

The training process:
- Uses a deep Q-network (DQN) to learn optimal game strategies
- Implements epsilon-greedy exploration to balance exploration and exploitation
- Saves models at score milestones (every 10 points) and keeps the best model
- Displays real-time training statistics

Training will continue indefinitely until you close the window. The longer you train, the better the AI will become.

### Playing Against the AI

After training, you can play against the AI:

```
python3 play.py
```

Game controls:
- **Arrow keys**: Control your snake (green)
- **R**: Restart the game after it ends
- Close the window to exit the game

In versus mode:
- The AI (blue) plays on the left side
- You (green) play on the right side
- The winner is determined by the highest score if both players die
- Each player can only eat food on their side of the screen

## How It Works

### Q-Learning Implementation

The AI uses a deep Q-network with the following features:
- Input layer: 11 nodes representing game state (danger directions, current direction, food location)
- Hidden layers: 256 nodes with ReLU activation
- Output layer: 3 actions (move straight, turn right, turn left)

The reward system:
- +10 for eating food
- -10 for collision (game over)
- +1 for moving closer to food
- No penalty for regular movement

### Model Architecture

```
QNetwork(
  (fc1): Linear(in_features=11, out_features=256)
  (fc2): Linear(in_features=256, out_features=256)
  (fc3): Linear(in_features=256, out_features=3)
)
```

## Project Structure

- `train.py`: Training script for the AI agent
- `play.py`: Human vs. AI gameplay
- `models/`: Directory storing trained models
  - `best_model.pth`: The best performing model
  - `model_XX.pth`: Models saved at score milestones

## Tips for Training

- Let the model train for at least 100 games to see basic competence
- For best results, train for 1000+ games
- Higher scores indicate better model performance
- If training is too slow, you can increase the `SPEED` constant in the code

## Customization

You can modify various parameters in the code:
- Game speed
- Board size
- Learning rate
- Reward values
- Network architecture

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses PyTorch for deep learning capabilities
- Game visualization is powered by Pygame
