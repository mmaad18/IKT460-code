from collections import defaultdict
import numpy as np


class MonteCarloControl:
    def __init__(self, action_space_size: int, gamma=0.99, epsilon=0.1):
        self.Q = defaultdict(lambda: np.zeros(action_space_size, dtype=np.float32))
        self.returns = defaultdict(list)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space_size = action_space_size


    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_size)  # Random action
        else:
            return np.argmax(self.Q[state])  # Greedy action


    def update_from_episode(self, episode):
        G = 0
        visited = set()

        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            if (state, action) not in visited:
                self.returns[(state, action)].append(G)
                self.Q[state][action] = np.mean(self.returns[(state, action)])
                visited.add((state, action))




