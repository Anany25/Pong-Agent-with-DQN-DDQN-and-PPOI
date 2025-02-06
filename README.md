
# Advanced Strategies in Pong: Reinforcement Learning

## Project Overview
This GitHub repository houses our project focused on mastering the Atari game Pong using advanced Reinforcement Learning (RL) techniques. We explored three main RL algorithms: Deep Q-Networks (DQN), Double Deep Q-Networks (DDQN), and Proximal Policy Optimization (PPO), to develop an autonomous agent that can outperform the game's built-in AI.

## Project Objectives
- **Development**: Implement RL agents using DQN, DDQN, and PPO algorithms.
- **Comparative Analysis**: Assess the algorithms based on learning speed, strategic efficiency, and gameplay performance.
- **Algorithm Enhancement**: Optimize these methods to boost performance in the Pong environment.

## Environment Setup
The project is built using the Pong environment from Gymnasium, which provides a realistic simulation of the classic Atari game.

### Prerequisites
Ensure you have Python installed along with the following libraries:
- Gym
- NumPy
- PyTorch
- Matplotlib

## Methodologies
Each RL method utilized different strategies and network architectures:
- **DQN and DDQN**: Employ an epsilon-greedy strategy, experience replay, and separate strategy/target networks.
- **PPO**: Uses an actor-critic architecture, multiple environment parallelism, and a clipping mechanism to update policies effectively.

## Results and Evaluation
The agents' performance improved over time, demonstrating the potential of each algorithm. Detailed findings, including learning curves and evaluation metrics, are available in the [results](results.md) section.

## Conclusion
Our study shows that while DQN provided a solid baseline, PPO offered the most substantial improvements in terms of robustness and efficiency. DDQN also showed promising results, especially in stability and performance.

## References
A comprehensive list of references can be found in the [references.md](references.md) file, providing context and background for the methods used.

## License
This project is open-sourced under the MIT license. See the [LICENSE](LICENSE.md) file for more details.

## Acknowledgments
- Contributions and data provided by Gymnasium.
- Theoretical foundations inspired by seminal papers and implementations in the field of RL.

For detailed academic references and methodologies, please refer to the literature listed in the project report.
