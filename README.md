# pacman-ai-project
This project simulates the Pacman game, offering two algorithms to control Pacman: Minimax or Genetic Algorithm (GA). Pacman navigates a grid to collect coins while avoiding the ghost, which minimizes Pacman’s score. The goal is to maximize the score through coin collection, bonus points, and hidden points over multiple rounds.

` This is my " Sophomore Final Project " `

### Welcome Page

#### Choose the Options:
Players can choose between implementing the Pacman game with either the **"Minimax"** or **"Genetic"** algorithm.

### Game Start

Pressing **Enter** starts the game.

### Explanation

#### (Option: Minimax)

- **Pacman** is the maximizing player, while the **ghost** is the minimizing player.
- Each agent aims to maximize their score to win, using the **minimax algorithm** influenced by the utility function.
- The **utility function** is calculated from the evaluation function provided in the code.

#### (Option: Genetic Algorithm (GA))

- Each agent focuses solely on maximizing their score by collecting bonus points obtained from successive coin combos.
- Time may be taken to reach individual coins, as agents prioritize successive coins.
- The game ends when agents approach the last remaining coins.

### Rules

- **Pacman** moves first, followed by the **ghost**. Scores are displayed on the right-hand side.
- Each solid coin collected earns 1 point.
- While playing, the terminal displays:
  - **"Bonus point:"**: Calculated from consecutive solid coin combos (including transparent coins). The score is equal to `combo_length^2`.
  - **"Hidden point"**: Occurs when a transparent coin is consumed, which may randomly transition to a solid coin with a probability of 0 - 0.5 (50%).
- Each round concludes when no solid coins remain. The game comprises **three rounds**.
- **Winners** of each round earn an additional **500 points**.

### Optional Requirements

1. **Probabilistic Minimax**: The behavior of coins introduces probabilistic elements, making the game non-deterministic. This randomness affects Pacman's decisions through the `scoreMinimax()` function.
  
2. **Stochastic Search (Genetic Algorithm)**: The genetic algorithm focuses on finding the best action sequence to maximize bonus points from successive coins, prioritizing efficient collection.

### ASCII Art (End)

