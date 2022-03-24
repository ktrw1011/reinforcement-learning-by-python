import time
import gym
from gym.wrappers.record_video import RecordVideo

from gym.envs.registration import register
register(id="FrozenLakeEasy-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"is_slippery": False})

if __name__ == "__main__":
    env = gym.make("FrozenLakeEasy-v0")
    # env = RecordVideo(env, "./video")
    s = env.reset()

    done = False
    while not done:
        print(env.render(mode='ascii'))
        time.sleep(1)
        n_state, reward, done, info = env.step(1)
        print("step")

    env.close()