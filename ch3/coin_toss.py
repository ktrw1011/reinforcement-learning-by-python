import random
from enum import Enum
from typing import List
import numpy as np

class Experience(Enum):
    # 探索
    exploration = 0
    # 活用
    exploitation = 1

def print_debug(text, flg):
    if flg:
        return print(text)

class CoinToss:
    def __init__(self, head_probs:List[float], max_episode_steps:int=30, verbose:bool=False) -> None:
        self.head_probs = head_probs
        self.max_episode_steps = max_episode_steps
        self.toss_count = 0
        self.verbose = verbose

    def __len__(self):
        return len(self.head_probs)

    def reset(self):
        self.toss_count = 0

    def step(self, action):
        final = self.max_episode_steps - 1
        if self.toss_count > final:
            raise Exception("トスカウントが最大エピソード数を超えました")

        else:
            done = self.toss_count == final

            if action >= len(self.head_probs):
                raise Exception(f"The No.{action} Coin Doesn't Exist.")
            
            head_prob = self.head_probs[action]
            if random.random() < head_prob:
                reward = 1.0
            else:
                reward = 0.0

            self.toss_count += 1
        
        return reward, done

class EpsilonGreedyAgent:
    def __init__(self, epsilon:float, verbose:bool=False) -> None:
        self.epsilon = epsilon    
        self.V = []
        self.verbose = verbose
        self.experiences = []

    def policy(self) -> int:
        """トスするコインを選ぶ"""
        coins = range(len(self.V))

        print_debug(f"現在のコインを投げたときの期待値は {self.V}", self.verbose)
         
        if random.random() < self.epsilon:
            # ランダムの値でepsilon以下なら、ランダムにコインを投げる
            selected_coin_idx =  random.choice(coins)
            print_debug(f"【探索します】", self.verbose)
            self.experiences.append(Experience.exploration)
        else:
            # epsilon以上なら現在最も期待値が大きい値のコインを投げる
            selected_coin_idx = int(np.argmax(self.V))
            self.experiences.append(Experience.exploitation)
            print_debug(f"【活用します】", self.verbose)

        return selected_coin_idx

    def play(self, env:CoinToss):
        # コインを選んだ回数
        N = [0] * len(env)
        # コインを投げて報酬を受け取れる期待値
        self.V = [0.] * len(env)

        env.reset()
        done = False
        rewards = []
        while not done:
            print_debug("="*10, self.verbose)
            # ポリシー(epsilon)に従ってトスするコインを選ぶ
            selected_coin = self.policy()
            print_debug(f"投げるコインは {env.head_probs[selected_coin]}", self.verbose)
            
            # コインを投げてリワード履歴に追加
            reward, done = env.step(selected_coin)
            print_debug(f"獲得したリワード {reward}", self.verbose)

            rewards.append(reward)

            n = N[selected_coin]
            coin_average = self.V[selected_coin]
            
            # 期待値を計算
            new_average = (coin_average * n + reward) / (n + 1)

            N[selected_coin] += 1

            # 各コイントスで得られる期待値を計算
            self.V[selected_coin] = new_average
            print_debug("="*10, self.verbose)
        
        return rewards, N