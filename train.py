import argparse
import difflib
import importlib
import os
import time
import uuid
import warnings
from collections import OrderedDict
from copy import deepcopy
from pprint import pprint

import gym
import numpy as np
import seaborn
import torch as th
import yaml
from stable_baselines3.common.buffers import NstepReplayBuffer  # noqa: F401
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike  # noqa: F401
from stable_baselines3.common.utils import constant_fn, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize, VecTransposeImage

try:
    from d3rlpy.algos import AWAC, AWR, BC, BCQ, BEAR, CQL
    from d3rlpy.sb3.convert import to_mdp_dataset
    from d3rlpy.sb3.wrappers import SB3Wrapper

    offline_algos = dict(awr=AWR, awac=AWAC, bc=BC, bcq=BCQ, bear=BEAR, cql=CQL)
except ImportError:
    offline_algos = {}

# For custom activation fn
from torch import nn as nn  # noqa: F401 pytype: disable=unused-import

# Register custom envs
import utils.import_envs  # noqa: F401 pytype: disable=import-error
from utils import ALGOS, get_latest_run_id, get_wrapper_class, linear_schedule, make_env
from utils.callbacks import SaveVecNormalizeCallback
from utils.hyperparams_opt import hyperparam_optimization
from utils.noise import LinearNormalActionNoise
from utils.utils import StoreDict, evaluate_policy_add_to_buffer, get_callback_class

seaborn.set()

if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument(
        "--offline-algo", help="Offline RL Algorithm", type=str, required=False, choices=list(offline_algos.keys())
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment ID")
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="", type=str)
    parser.add_argument("-i", "--trained-agent", help="Path to a pretrained agent to continue training", default="", type=str)
    parser.add_argument("-n", "--n-timesteps", help="Overwrite the number of timesteps", default=-1, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--log-interval", help="Override log interval (default: -1, no change)", default=-1, type=int)
    parser.add_argument(
        "--eval-freq", help="Evaluate the agent every n steps (if negative, no evaluation)", default=10000, type=int
    )
    parser.add_argument("--eval-episodes", help="Number of episodes to use for evaluation", default=5, type=int)
    parser.add_argument("--save-freq", help="Save the model every n steps (if negative, no checkpoint)", default=-1, type=int)
    parser.add_argument(
        "--save-replay-buffer", help="Save the replay buffer too (when applicable)", action="store_true", default=False
    )
    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="logs")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    parser.add_argument("--n-trials", help="Number of trials for optimizing hyperparameters", type=int, default=10)
    parser.add_argument(
        "-optimize", "--optimize-hyperparameters", action="store_true", default=False, help="Run hyperparameters search"
    )
    parser.add_argument("--n-jobs", help="Number of parallel jobs when optimizing hyperparameters", type=int, default=1)
    parser.add_argument(
        "--sampler",
        help="Sampler to use when optimizing hyperparameters",
        type=str,
        default="tpe",
        choices=["random", "tpe", "skopt"],
    )
    parser.add_argument(
        "--pruner",
        help="Pruner to use when optimizing hyperparameters",
        type=str,
        default="median",
        choices=["halving", "median", "none"],
    )
    parser.add_argument("--n-startup-trials", help="Number of trials before using optuna sampler", type=int, default=10)
    parser.add_argument("--n-evaluations", help="Number of evaluations for hyperparameter optimization", type=int, default=20)
    parser.add_argument(
        "--storage", help="Database storage path if distributed optimization should be used", type=str, default=None
    )
    parser.add_argument("--study-name", help="Study name for distributed optimization", type=str, default=None)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "-params",
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",
    )
    parser.add_argument("-b", "--pretrain-buffer", help="Path to a saved replay buffer for pretraining", type=str)
    parser.add_argument(
        "--pretrain-params",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Optional arguments for pretraining with replay buffer",
    )
    parser.add_argument("-uuid", "--uuid", action="store_true", default=False, help="Ensure that the run has a unique ID")
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())  # pytype: disable=module-attr

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(f"{env_id} not found in gym registry, you maybe meant {closest_match}?")

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = f"_{uuid.uuid4()}" if args.uuid else ""
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2 ** 32 - 1, dtype="int64").item()

    set_random_seed(args.seed)

    # Setting num threads to 1 makes things run faster on cpu
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"

    tensorboard_log = None if args.tensorboard_log == "" else os.path.join(args.tensorboard_log, env_id)

    is_atari = False
    if "NoFrameskip" in env_id:
        is_atari = True

    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {args.seed}")

    # Load hyperparameters from yaml file
    with open(f"hyperparams/{args.algo}.yml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        elif is_atari:
            hyperparams = hyperparams_dict["atari"]
        else:
            raise ValueError(f"Hyperparameters not found for {args.algo}-{env_id}")

    if args.hyperparams is not None:
        # Overwrite hyperparams if needed
        hyperparams.update(args.hyperparams)
    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
    env_kwargs = {} if args.env_kwargs is None else args.env_kwargs

    algo_ = args.algo
    # HER is only a wrapper around an algo
    if args.algo == "her":
        algo_ = saved_hyperparams["model_class"]
        assert algo_ in {"sac", "ddpg", "dqn", "td3"}, "{} is not compatible with HER".format(algo_)
        # Retrieve the model class
        hyperparams["model_class"] = ALGOS[saved_hyperparams["model_class"]]

    if args.verbose > 0:
        pprint(saved_hyperparams)

    n_envs = hyperparams.get("n_envs", 1)

    if args.verbose > 0:
        print(f"Using {n_envs} environments")

    # Create schedules
    for key in ["learning_rate", "clip_range", "clip_range_vf"]:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split("_")
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], (float, int)):
            # Negative value: ignore (ex: for clipping)
            if hyperparams[key] < 0:
                continue
            hyperparams[key] = constant_fn(float(hyperparams[key]))
        else:
            raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")

    # Should we overwrite the number of timesteps?
    if args.n_timesteps > 0:
        if args.verbose:
            print(f"Overwriting n_timesteps with n={args.n_timesteps}")
        n_timesteps = args.n_timesteps
    else:
        n_timesteps = int(hyperparams["n_timesteps"])

    normalize = False
    normalize_kwargs = {}
    if "normalize" in hyperparams.keys():
        normalize = hyperparams["normalize"]
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams["normalize"]

    for key in ["replay_buffer_kwargs", "policy_kwargs", "replay_buffer_class"]:
        if key in hyperparams.keys():
            # Convert to python object if needed
            if isinstance(hyperparams[key], str):
                hyperparams[key] = eval(hyperparams[key])

    # Delete keys so the dict can be pass to the model constructor
    if "n_envs" in hyperparams.keys():
        del hyperparams["n_envs"]
    del hyperparams["n_timesteps"]

    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)
    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    log_path = f"{args.log_folder}/{args.algo}/"
    save_path = os.path.join(log_path, f"{env_id}_{get_latest_run_id(log_path, env_id) + 1}{uuid_str}")
    params_path = f"{save_path}/{env_id}"
    os.makedirs(params_path, exist_ok=True)

    callbacks = get_callback_class(hyperparams)
    if "callback" in hyperparams.keys():
        del hyperparams["callback"]

    if args.save_freq > 0:
        # Account for the number of parallel environments
        args.save_freq = max(args.save_freq // n_envs, 1)
        callbacks.append(CheckpointCallback(save_freq=args.save_freq, save_path=save_path, name_prefix="rl_model", verbose=1))

    def create_env(n_envs, eval_env=False, no_log=False):
        """
        Create the environment and wrap it if necessary
        :param n_envs: (int)
        :param eval_env: (bool) Whether is it an environment used for evaluation or not
        :param no_log: (bool) Do not log training when doing hyperparameter optim
            (issue with writing the same file)
        :return: (Union[gym.Env, VecEnv])
        """
        global hyperparams
        global env_kwargs

        # Do not log eval env (issue with writing the same file)
        log_dir = None if eval_env or no_log else save_path

        if n_envs == 1:
            env = DummyVecEnv(
                [make_env(env_id, 0, args.seed, wrapper_class=env_wrapper, log_dir=log_dir, env_kwargs=env_kwargs)]
            )
        else:
            # env = SubprocVecEnv([make_env(env_id, i, args.seed) for i in range(n_envs)])
            # On most env, SubprocVecEnv does not help and is quite memory hungry
            env = DummyVecEnv(
                [
                    make_env(env_id, i, args.seed, log_dir=log_dir, env_kwargs=env_kwargs, wrapper_class=env_wrapper)
                    for i in range(n_envs)
                ]
            )
        if normalize:
            # Copy to avoid changing default values by reference
            local_normalize_kwargs = normalize_kwargs.copy()
            # Do not normalize reward for env used for evaluation
            if eval_env:
                if len(local_normalize_kwargs) > 0:
                    local_normalize_kwargs["norm_reward"] = False
                else:
                    local_normalize_kwargs = {"norm_reward": False}

            if args.verbose > 0:
                if len(local_normalize_kwargs) > 0:
                    print(f"Normalization activated: {local_normalize_kwargs}")
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **local_normalize_kwargs)

        # Optional Frame-stacking
        if hyperparams.get("frame_stack", False):
            n_stack = hyperparams["frame_stack"]
            env = VecFrameStack(env, n_stack)
            print(f"Stacking {n_stack} frames")

        if is_image_space(env.observation_space):
            if args.verbose > 0:
                print("Wrapping into a VecTransposeImage")
            env = VecTransposeImage(env)
        return env

    env = create_env(n_envs)

    # Create test env if needed, do not normalize reward
    eval_env = None
    if args.eval_freq > 0 and not args.optimize_hyperparameters:
        # Account for the number of parallel environments
        args.eval_freq = max(args.eval_freq // n_envs, 1)

        if "NeckEnv" in env_id:
            # Use the training env as eval env when using the neck
            # because there is only one robot
            # there will be an issue with the reset
            eval_callback = EvalCallback(
                env, callback_on_new_best=None, best_model_save_path=save_path, log_path=save_path, eval_freq=args.eval_freq
            )
            callbacks.append(eval_callback)
        else:
            if args.verbose > 0:
                print("Creating test environment")

            save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=params_path)
            eval_callback = EvalCallback(
                create_env(1, eval_env=True),
                callback_on_new_best=save_vec_normalize,
                best_model_save_path=save_path,
                n_eval_episodes=args.eval_episodes,
                log_path=save_path,
                eval_freq=args.eval_freq,
                deterministic=not is_atari,
            )

            callbacks.append(eval_callback)

    # TODO: check for hyperparameters optimization
    # TODO: check What happens with the eval env when using frame stack
    if "frame_stack" in hyperparams:
        del hyperparams["frame_stack"]

    # Stop env processes to free memory
    if args.optimize_hyperparameters and n_envs > 1:
        env.close()

    # Parse noise string for DDPG and SAC
    if algo_ in ["ddpg", "sac", "td3"] and hyperparams.get("noise_type") is not None:
        noise_type = hyperparams["noise_type"].strip()
        noise_std = hyperparams["noise_std"]
        n_actions = env.action_space.shape[0]
        if "normal" in noise_type:
            if "lin" in noise_type:
                final_sigma = hyperparams.get("noise_std_final", 0.0) * np.ones(n_actions)
                hyperparams["action_noise"] = LinearNormalActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=noise_std * np.ones(n_actions),
                    final_sigma=final_sigma,
                    max_steps=n_timesteps,
                )
            else:
                hyperparams["action_noise"] = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
        elif "ornstein-uhlenbeck" in noise_type:
            hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
            )
        else:
            raise RuntimeError(f'Unknown noise type "{noise_type}"')
        print(f"Applying {noise_type} noise with std {noise_std}")
        del hyperparams["noise_type"]
        del hyperparams["noise_std"]
        if "noise_std_final" in hyperparams:
            del hyperparams["noise_std_final"]

    if args.trained_agent.endswith(".zip") and os.path.isfile(args.trained_agent):
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del hyperparams["policy"]

        if "policy_kwargs" in hyperparams.keys():
            del hyperparams["policy_kwargs"]

        model = ALGOS[args.algo].load(
            args.trained_agent, env=env, seed=args.seed, tensorboard_log=tensorboard_log, verbose=args.verbose, **hyperparams
        )

        exp_folder = args.trained_agent.split(".zip")[0]
        if normalize:
            print("Loading saved running average")
            stats_path = os.path.join(exp_folder, env_id)
            if os.path.exists(os.path.join(stats_path, "vecnormalize.pkl")):
                env = VecNormalize.load(os.path.join(stats_path, "vecnormalize.pkl"), env)
            else:
                # Legacy:
                env.load_running_average(exp_folder)

        replay_buffer_path = os.path.join(os.path.dirname(args.trained_agent), "replay_buffer.pkl")
        if os.path.exists(replay_buffer_path):
            print("Loading replay buffer")
            model.load_replay_buffer(replay_buffer_path)

    elif args.optimize_hyperparameters:

        if args.verbose > 0:
            print("Optimizing hyperparameters")

        if args.storage is not None and args.study_name is None:
            warnings.warn(
                f"You passed a remote storage: {args.storage} but no `--study-name`."
                "The study name will be generated by Optuna, make sure to re-use the same study name "
                "when you want to do distributed hyperparameter optimization."
            )

        def create_model(*_args, **kwargs):
            """
            Helper to create a model with different hyperparameters
            """
            return ALGOS[args.algo](env=create_env(n_envs, no_log=True), tensorboard_log=tensorboard_log, verbose=0, **kwargs)

        data_frame = hyperparam_optimization(
            args.algo,
            create_model,
            create_env,
            n_trials=args.n_trials,
            n_timesteps=n_timesteps,
            hyperparams=hyperparams,
            n_jobs=args.n_jobs,
            seed=args.seed,
            sampler_method=args.sampler,
            pruner_method=args.pruner,
            n_startup_trials=args.n_startup_trials,
            n_evaluations=args.n_evaluations,
            storage=args.storage,
            study_name=args.study_name,
            verbose=args.verbose,
            deterministic_eval=not is_atari,
        )

        report_name = (
            f"report_{env_id}_{args.n_trials}-trials-{n_timesteps}" f"-{args.sampler}-{args.pruner}_{int(time.time())}.csv"
        )

        log_path = os.path.join(args.log_folder, args.algo, report_name)

        if args.verbose:
            print(f"Writing report to {log_path}")

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        data_frame.to_csv(log_path)
        exit()
    else:
        # Train an agent from scratch
        model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, seed=args.seed, verbose=args.verbose, **hyperparams)

    kwargs = {}
    if args.log_interval > -1:
        kwargs = {"log_interval": args.log_interval}

    if len(callbacks) > 0:
        kwargs["callback"] = callbacks

    # Save hyperparams
    with open(os.path.join(params_path, "config.yml"), "w") as f:
        yaml.dump(saved_hyperparams, f)

    # save command line arguments
    with open(os.path.join(params_path, "args.yml"), "w") as f:
        ordered_args = OrderedDict([(key, vars(args)[key]) for key in sorted(vars(args).keys())])
        yaml.dump(ordered_args, f)

    print(f"Log path: {save_path}")

    if args.pretrain_buffer is not None:
        # n_step_replay = args.pretrain_params.get("n_step_replay", 1)
        # print(f"n_step_replay={n_step_replay}")
        # old_buffer = deepcopy(model.replay_buffer)
        # old_buffer = model.replay_buffer

        # n_step_buffer = NstepReplayBuffer(
        #     old_buffer.buffer_size,
        #     old_buffer.observation_space,
        #     old_buffer.action_space,
        #     old_buffer.device,
        #     gamma=model.gamma,
        #     n_step=n_step_replay,
        # )
        # n_step_buffer.actor = model.actor
        # n_step_buffer.ent_coef = 0.0
        model.load_replay_buffer(args.pretrain_buffer)

        # pos = model.replay_buffer.size()
        # Load expert data
        # TODO: keep old data
        # n_step_buffer.extend(
        #     model.replay_buffer.observations[:pos],
        #     model.replay_buffer.next_observations[:pos],
        #     model.replay_buffer.actions[:pos],
        #     model.replay_buffer.rewards[:pos],
        #     model.replay_buffer.dones[:pos],
        # )
        # model.replay_buffer = n_step_buffer
        print(f"Buffer size = {model.replay_buffer.buffer_size}")
        # Artificially reduce buffer size
        # model.replay_buffer.full = False
        # model.replay_buffer.pos = 5000

        print(f"{model.replay_buffer.size()} transitions in the replay buffer")

        n_iterations = args.pretrain_params.get("n_iterations", 10)
        n_epochs = args.pretrain_params.get("n_epochs", 1)
        q_func_type = args.pretrain_params.get("q_func_type")
        batch_size = args.pretrain_params.get("batch_size", 512)
        # n_action_samples = args.pretrain_params.get("n_action_samples", 1)
        n_eval_episodes = args.pretrain_params.get("n_eval_episodes", 5)
        add_to_buffer = args.pretrain_params.get("add_to_buffer", False)
        deterministic = args.pretrain_params.get("deterministic", True)
        for arg_name in {
            "n_iterations",
            "n_epochs",
            "q_func_type",
            "batch_size",
            "n_eval_episodes",
            "add_to_buffer",
            "deterministic",
        }:
            if arg_name in args.pretrain_params:
                del args.pretrain_params[arg_name]
        try:
            assert args.offline_algo is not None and offline_algos is not None
            kwargs_ = {} if q_func_type is None else dict(q_func_type=q_func_type)
            kwargs_.update(args.pretrain_params)

            offline_model = offline_algos[args.offline_algo](
                n_epochs=n_epochs,
                batch_size=batch_size,
                **kwargs_,
            )
            offline_model = SB3Wrapper(offline_model)
            offline_model.use_sde = False
            # break the logger...
            # offline_model.replay_buffer = model.replay_buffer

            for i in range(n_iterations):
                dataset = to_mdp_dataset(model.replay_buffer)
                offline_model.fit(dataset.episodes)

                mean_reward, std_reward = evaluate_policy_add_to_buffer(
                    offline_model,
                    model.get_env(),
                    n_eval_episodes=n_eval_episodes,
                    replay_buffer=model.replay_buffer if add_to_buffer else None,
                    deterministic=deterministic,
                )
                print(f"Iteration {i + 1} training, mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                if mean_reward > 2000 and i > 10:
                    # break
                    # add_to_buffer = True
                    # deterministic = False
                    # exp_temperature = 1.0
                    # print("Adding to buffer")
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            print("Starting training")

    try:
        model.learn(n_timesteps, eval_log_path=save_path, eval_env=eval_env, eval_freq=args.eval_freq, **kwargs)
    except KeyboardInterrupt:
        pass
    finally:
        # Release resources
        env.close()

    # Save trained model

    print(f"Saving to {save_path}")
    model.save(f"{save_path}/{env_id}")

    if hasattr(model, "save_replay_buffer") and args.save_replay_buffer:
        print("Saving replay buffer")
        model.save_replay_buffer(os.path.join(save_path, "replay_buffer.pkl"))

    if normalize:
        # Important: save the running average, for testing the agent we need that normalization
        model.get_vec_normalize_env().save(os.path.join(params_path, "vecnormalize.pkl"))
        # Deprecated saving:
        # env.save_running_average(params_path)
