import numpy as np

class ELAgent:
    def __init__(self, epsilon:float) -> None:
        self.Q =  {}
        self.epsilon = epsilon
        self.reward_log = []

    def policy(self, s, actions):

        # epsilonに応じた「活用」と「探索」の選択
        if np.random.random() < self.epsilon:
            return np.random.randint(len(actions))

        if s in self.Q and sum(self.Q[s]) != 0:
            return np.argmax(self.Q[s])

        else:
            return np.random.randint(len(actions))

    def init_log(self) -> None:
        self.reward_log = []

    def log(self, reward) -> None:
        self.reward_log.append(reward)

    def show_reward_log(self, interval:int=50, episode:int=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.mean(rewards)
            std = np.std(rewards)
            print(f"At Episode: {episode} average reward is {mean:3.f} (+/- {std:3f})")
        
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i+interval)]
                means.append(np.mean(rewards))
                stds.append(np.mean(stds))
            
            means = np.array(means)
            stds = np.array(stds)