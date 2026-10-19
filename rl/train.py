import numpy as np
from dataclasses import dataclass
import os
import pathlib
import sys
import time

directory = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(directory))

from env import make_env
from models import Qtable
from utils.logger import Logger
from collections import OrderedDict


@dataclass
class EnvConfig:
    domain: str = "double_pendulum"
    task: str = "balance"
    repeat: int = 2
    num_digitized: int = 16
    num_action: int = 9
    state_size: int = num_digitized**4
    gamma: float = 0.99
    alpha: float = 0.5
    max_episode: int = int(10e7)
    episode_length: int = 200
    should_log_model: int = 10000
    should_log_scalar: int = 200
    should_log_video: int = 1000
    restore: bool = False
    restore_path: str = ""
    video_length: int = 200
    logdir: str = pathlib.Path().joinpath("./logs/train", str(time.strftime("%m-%d-%H-%M-%S")))


class Agent():
    def __init__(self, config: EnvConfig) -> None:
        self._config = config
        self._build_model()

    def get_action(self, state, global_step=None, explore=True, method="softmax"):
        return self._qtable.get_action(state, explore, global_step, method=method)

    def update_Qtable(self, state, action, reward, next_state):
        return self._qtable.update_Qtable(state, action, reward, next_state)

    def _build_model(self):
        self._qtable = Qtable(self._config)
    
def main():
    config = EnvConfig()
    env = make_env(config)
    os.makedirs(config.logdir, exist_ok=True)
    print(f'log: {config.logdir}')
    logger = Logger(config.logdir)
    agent = Agent(config)

    if config.restore:
        print(config.restore_path)
        qtable = np.load(config.restore_path)
        agent._qtable._Qtable = qtable

    # ランダムな行動を使ってQtableを初期化
    for i in range(100):
        state = env.reset()
        for step in range(400):
            action = agent.get_action(state, explore=True, method="random")
            next_state, reward, done, _ = env.step(action)
            qtable = agent.update_Qtable(state, action, reward, next_state)
            state = next_state
            if done:
                break

    ave_ep_len = []
    prev_q = agent._qtable._Qtable.copy()
    total_time = 0
    # main training loop
    for episode in range(config.max_episode):
        # １エピソードの実行
        start_time = time.time()
        state = env.reset()
        episode_reward = 0
        best_num = 0
        for step in range(config.episode_length):
            action = agent.get_action(state, logger.global_step, method='epsilon-greedy')
            next_state, reward, done, state_dict = env.step(action)
            qtable = agent.update_Qtable(state, action, reward, next_state)
            episode_reward += reward
            best_num += state_dict["best"]
            state = next_state
            if done:
                break
        ave_ep_len.append(step)
        # スカラーをログに保存
        if episode % config.should_log_scalar == 0:
            print(f"episode: {episode}, episode_reward: {episode_reward}, episode_steps: {step+1}")
            qtable_err = np.mean(np.abs((agent._qtable._Qtable - prev_q)))
            prev_q = agent._qtable._Qtable.copy()
            logger.add_scalars(OrderedDict([
                ("train/ep_reward", episode_reward),
                ("train/ep_length", step),
                ("train/ep_best_num", best_num),
                (f"train/ave_ep_len_{config.should_log_scalar}", sum(ave_ep_len)/len(ave_ep_len)),
                ("train/qtable_error", qtable_err),
                ("time/total_minute", total_time / 60)
            ]))
            ave_ep_len = []

        # qtableを保存
        if episode % config.should_log_model == 0 and episode != 0:
            save_file = config.logdir.joinpath(f"qtable_{episode}.npy")
            np.save(save_file, qtable)
            print(f'\nsave model {save_file}\n')

        # 評価と動画の保存
        if episode % config.should_log_video == 0 and episode != 0:
            state = env.reset()
            eval_reward = 0
            best_num = 0
            img_seq = []
            act_seq = []
            img_seq = []
            rew_seq = []
            alpha_seq = []
            theta_seq = []
            for _ in range(config.video_length):
                img_seq.append(env._env.physics.render(height=480, width=640,camera_id=0))
                action = (agent.get_action(state, explore=False))
                next_state, reward, done, state_dict = env.step(action)
                state = next_state
                act_seq.append(env._digitized_action[action])
                rew_seq.append(reward)
                alpha_seq.append(state_dict['pendulum_rad'])
                theta_seq.append(state_dict['arm_rad'])
                eval_reward += reward
                best_num += state_dict["best"]
                if done:
                    break
            print(f"\nevaluate episode reward: {eval_reward}, episode step: {len(rew_seq)}\n")
            logger.log_video({'eval/video': np.array(img_seq)}, save=False)
            logger.add_scalars(OrderedDict([
                ("eval/ep_reward", eval_reward),
                ("eval/ep_best_num", best_num)
                ]))
            act_seq_img = logger.plot2image('action', {'torque': np.array(act_seq)})
            rew_seq_img = logger.plot2image('reward', {'reward': np.array(rew_seq)})
            state_seq_img = logger.plot2image('state', {'alpha': np.array(alpha_seq),
                                                    'theta': np.array(theta_seq)})
            img_dict = {'eval/action': act_seq_img,
                        'eval/reward': rew_seq_img,
                        'eval/state': state_seq_img}
            logger.log_image(img_dict)
        total_time += time.time() - start_time
        start_time = time.time
        logger.step()

if __name__ == "__main__":
    main()

