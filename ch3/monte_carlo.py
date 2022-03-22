import math
from collections import defaultdict
from typing import Dict, List

import gym
from el_agent import ELAgent

class MonteCarloAgent(ELAgent):
    
    def __init__(self, epsilon: float=0.1) -> None:
        super().__init__(epsilon)

    def learn(self, env:gym.Env, episode_count:int=1000, gamma:float=0.9, render=False, report_interval:int=50):
        self.init_log()
        # n=4 [0, 1, 2, 3]
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0]*len(actions))
        N = defaultdict(lambda: [0]*len(actions))

        for e in range(episode_count):
            s = env.reset()
            done = False
            experience = []

            # エピソードが終わるまでプレイする
            while not done:
                if render:
                    env.render()

                a = self.policy(s, actions)
                n_state, reward, done, info = env.step(a)
                experience.append({"state":s, "action":a, "reward":reward})
                s = n_state
            else:
                self.log(reward)

            
            for i, ex in enumerate(experience):
                s, a = ex["state"], ex["action"]

                G, t = 0, 0
                for j in range(i, len(experience)):
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1

                N[s][a] += 1
                alpha = 1/ N[s][a]
                self.Q[s][a] += alpha * (G - self.Q[s][a])

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)

            


