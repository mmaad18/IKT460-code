

class DQNAgent:
    def __init__(
            self,
            state_size,
            action_size,
            learning_rate=0.001,
            discount_factor=0.99,
            epsilon_start=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            target_update_freq=100
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Initialize replay buffer, Q-network, and target network here