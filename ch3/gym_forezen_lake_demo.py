import streamlit as st
import gym
# from gym.envs.registration import register
# register(id="FrozenLakeEasy-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",
#          kwargs={"is_slippery": False})

action_name = {
    0: "左",
    1: "下",
    2: "右",
    3: "上"
}

def show_log(action, state, reward, done, info):
    st.write(f"action: {action_name[action]}, state: {state}, reward: {reward}, done: {done}, info: {info}")


btn = st.button("ランダムアクション")
st.write("ゴールにたどり付けば報酬1が獲得できる")

if "env" not in st.session_state:
    env = gym.make("FrozenLake", is_slippery=False)
    s  = env.reset()
    st.session_state["env"] = env
    img = env.render(mode='rgb_array')
    st.image(img)
    st.subheader("ログ")
else:
    env = st.session_state["env"]

if btn:
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(env.action_space.sample())
    img = env.render(mode='rgb_array')
    st.image(img)
    st.subheader("ログ")
    show_log(action, n_state, reward, done, info)