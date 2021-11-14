import multiprocessing
from typing import Callable

import gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from transformers import AutoTokenizer

from rl_reader.envs.sentpiece_mlm import SentPieceMLM


class TextSubprocVecEnv(SubprocVecEnv):
    def get_images(self):
        raise NotImplementedError

    def get_text(self):
        for pipe in self.remotes:
            # gather text from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "text"))
        texts = [pipe.recv() for pipe in self.remotes]
        return texts

    def render(self, *args, **kwargs):
        try:
            texts = self.get_text()
        except NotImplementedError:
            warnings.warn(f"Render not defined for {self}")
            return
        print(texts)


def load_docs(path, text_col='text'):
    df = pd.read_pickle(path)
    text = df[text_col]
    return text


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def make_env(docs, tokenizer, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = SentPieceMLM(docs, tokenizer)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


def main(args):
    docs = load_docs(args.corpus_path, text_col=args.text_col)
    tokenizer = load_tokenizer(args.model_name)

    env = TextSubprocVecEnv([
        make_env(docs, tokenizer, i) for i in range(args.num_procs)
    ])

    model = PPO("MlpPolicy", env, n_steps=args.num_steps, verbose=1)
    if args.logdir is not None:
        logger = configure(args.logdir, ["stdout", "tensorboard"])
        model.set_logger(logger)

    try:
        model.learn(total_timesteps=args.total_steps)
    except KeyboardInterrupt:
        pass
    env.close()

    env = SentPieceMLM(docs, tokenizer)
    try:
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')
    except KeyboardInterrupt:
        pass
    print(env.render())
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
    p.add_argument('--total_steps', default=1E8, type=int)
    args = p.parse_args()

    main(args)
