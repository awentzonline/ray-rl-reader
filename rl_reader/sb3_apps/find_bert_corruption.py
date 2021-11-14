import multiprocessing
from typing import Callable

import gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from transformers import AutoModel, AutoTokenizer

from rl_reader.envs.bert_corruption import BertCorruptionFinder


def load_docs(path, text_col='text'):
    df = pd.read_pickle(path)
    text = df[text_col]
    return text


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def make_env(docs, tokenizer, model, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = BertCorruptionFinder(docs, tokenizer, model)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


def main(args):
    docs = load_docs(args.corpus_path, text_col=args.text_col)
    tokenizer, model = load_model(args.model_name)
    model = model.eval()

    env = SubprocVecEnv([
        make_env(docs, tokenizer, model, i) for i in range(args.num_procs)
    ])

    model = PPO("MlpPolicy", env, n_steps=args.num_steps, verbose=1)
    if args.logdir is not None:
        logger = configure(args.logdir, ["stdout", "tensorboard"])
        model.set_logger(logger)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()

    env.close()


if __name__ == '__main__':
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument('corpus_path', help='Location of plaintext')
    p.add_argument(
        '--model_name', default='bert-base-uncased',
        help='Huggingface transformer model name'
    )
    p.add_argument('--text_col', default='text')
    p.add_argument('--logdir', default='runs')
    p.add_argument('--seed', default=1, type=int)
    p.add_argument('--num_procs', default=multiprocessing.cpu_count(), type=int)
    p.add_argument('--num_steps', default=1024, type=int)
    args = p.parse_args()

    main(args)
