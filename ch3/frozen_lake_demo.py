import time
import gym

if __name__ == "__main__":
    env = gym.make("FrozenLake", is_slippery=False)
    s = env.reset()

    done = False
    while not done:
        env.render()
        time.sleep(1)
        n_state, reward, done, info = env.step(env.action_space.sample())
        print("step")

    env.close()