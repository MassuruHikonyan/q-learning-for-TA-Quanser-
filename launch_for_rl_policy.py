import time
from dataclasses import dataclass
import pathlib
import numpy as np

from rl.models import Qtable
from utils.logger import Logger
from real.invpen import Invpen
from rl.env import make_env

@dataclass
class EnvConfig:
    domain: str = "double_pendulum"
    task: str = "swingup"  # 1. swingup タスクに変更
    repeat: int = 2
    num_digitized: int = 16
    num_action: int = 9
    state_size: int = num_digitized**4
    gamma: float = 0.99
    alpha: float = 0.5
    max_episode: int = int(10e7)
    episode_length: int = 500
    should_log_model: int = 10000
    should_log_scalar: int = 200
    should_log_video: int = 10000
    restore: bool = True
    # 2. 振り上げ用Qテーブルのパスを指定（環境に合わせて変更してください）
    restore_path: str = "logs/train/best_policy/qtable_40000.npy"
    video_length: int = 200
    logdir: str = pathlib.Path().joinpath("./logs/real-by-rl", str(time.strftime("%m-%d-%H-%M-%S")))


class TrainedAgent():
    def __init__(self):
        config = EnvConfig()
        self.config = config
        # pretrained qtable
        self.qtable = Qtable(config)
        self.logger = Logger(config.logdir)
        print(config.restore_path)
        self.qtable.load(config.restore_path)
        self.env = make_env(config)
        self.start = True  # 3. 初期状態から制御を開始させる
        self.data = {'n_alpha': [], 'n_theta': []}

    def _digitized_obs(self, obs):
        '''
        obs[0] theta
        obs[1] alpha
        obs[2] theta dot
        obs[3] alpha dot
        '''
        d = self.config.num_digitized
        n_arm_rad = np.digitize(obs[0],
                                np.linspace(-np.pi, np.pi, d//self.env._arm_div +1)[1:-1])
        n_arm_vel = np.digitize(obs[2].clip(-8, 8), np.linspace(-8, 8, d+1)[1:-1])

        n_pen_rad = np.digitize(obs[1],
                                np.linspace(-np.pi, np.pi, d+1)[1:-1] + self.env._arrange)
        n_pen_vel = np.digitize(obs[3].clip(-8, 8), np.linspace(-8, 8, d+1)[1:-1])

        state_dict = {}
        state_dict["n_arm_rad"] = n_arm_rad
        state_dict["n_pen_rad"] = n_pen_rad
        state_dict["digitized_state"] = n_pen_rad + n_pen_vel*d + n_arm_vel*d**2 + n_arm_rad*d**3
        arm_cond = n_arm_rad == 0 or n_arm_rad == d//self.env._arm_div - 1
        pen_cond = n_pen_rad == 0 or n_pen_rad == d -1
        if arm_cond or pen_cond:
            done = True
        else:
            done = False
        return state_dict["digitized_state"], done, state_dict

    def policy(self, obs):
        # 最初から RL ポリシーに従ってトルクを出力する
        digitized_state, done, state_dict = self._digitized_obs(obs)
        digitized_action = self.qtable.get_action(digitized_state, False)
        torque = self.env._digitized_action[digitized_action]
        
        self.data['n_alpha'].append(state_dict['n_pen_rad'])
        self.data['n_theta'].append(state_dict['n_arm_rad'])
        
        # 安全用リミッター（アームが端に行きすぎたらトルクオフ）
        if done or abs(obs[0]) > 0.8 * np.pi:
            return 0.0
            
        return torque

    def after_termination_func(self, data):
        data.update({'n_alpha': self.data['n_alpha'], 'n_theta': self.data['n_theta']})
        data = {key: np.array(value) for key, value in data.items()}
        state_seq_img = self.logger.plot2image('state', {'alpha': data['alpha'],
                                                    'theta': data['theta']})
        n_state_seq_img = self.logger.plot2image('n_state', {'n_alpha': data['n_alpha'],
                                                        'n_theta': data['n_theta']})
        statedot_seq_img = self.logger.plot2image('dot_state', {'alphadot': data['alphadot'],
                                                                'thetadot': data['thetadot']})
        act_seq_img = self.logger.plot2image('action', {'torque': data['torque']})
        time_seq_img = self.logger.plot2image('time', {'time': data['time']})
        dt_seq_img = self.logger.plot2image('dt', {'dt': data['dt']})
        alphaf_seq_img = self.logger.plot2image('alphaf', {'alphaf': data['alpha_f']})
        img_dict = {'action': act_seq_img,
                    'state/nodot': state_seq_img,
                    'state/dot': statedot_seq_img,
                    'state/alphaf': alphaf_seq_img,
                    'state/n': n_state_seq_img,
                    'time': time_seq_img,
                    'dt': dt_seq_img}
        self.logger.log_image(img_dict)

def main():
    agent = TrainedAgent()
    my_invpen = Invpen(agent)

    my_invpen.run(sample_time=0.015, simulation_time=60.0, figure=True, logging=True)

if __name__ == "__main__":
    main()