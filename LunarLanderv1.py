import gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.dqn.policies import MlpPolicy, CnnPolicy



env = gym.make('LunarLander-v2')
model = DQN("MlpPolicy", env, gamma=0.99, learning_rate=0.0005, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True, learning_starts=1000, target_network_update_freq=500, prioritized_replay=False, prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-06, param_noise=False, n_cpu_tf_sess=None, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None)
model.learn(total_timesteps=200000)

obs = env.reset()
for _ in range(1000):
	action, _ = model.predict(obs, deterministic=True)
	obs, reward, done, _ = env.step(action)
	print(obs)
	env.render()
	if done:
		break



# Set up fake display; otherwise rendering will fail
# import os
# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ['DISPLAY'] = ':1'

# import base64
# from pathlib import Path

# from IPython import display as ipythondisplay
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

# def show_videos(video_path='', prefix=''):
#   """
#   Taken from https://github.com/eleurent/highway-env

#   :param video_path: (str) Path to the folder containing videos
#   :param prefix: (str) Filter the video, showing only the only starting with this prefix
#   """
#   html = []
#   for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
#       video_b64 = base64.b64encode(mp4.read_bytes())
#       html.append('''<video alt="{}" autoplay 
#                     loop controls style="height: 400px;">
#                     <source src="data:video/mp4;base64,{}" type="video/mp4" />
#                 </video>'''.format(mp4, video_b64.decode('ascii')))
#   ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))

#   from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

# def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
#   """
#   :param env_id: (str)
#   :param model: (RL model)
#   :param video_length: (int)
#   :param prefix: (str)
#   :param video_folder: (str)
#   """
#   eval_env = DummyVecEnv([lambda: gym.make(env_id)])
#   # Start the video at step=0 and record 500 steps
#   eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
#                               record_video_trigger=lambda step: step == 0, video_length=video_length,
#                               name_prefix=prefix)

#   obs = eval_env.reset()
#   for _ in range(video_length):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, _, _, _ = eval_env.step(action)

#   # Close the video recorder
#   eval_env.close()

# record_video('LunarLander-v2', model, video_length=500, prefix='dqn-lunar_lander')
# show_videos('videos', prefix='dqn')


# https://stable-baselines.readthedocs.io/en/master/guide/examples.html