import multiprocessing

import gym
import ray
from ray import tune
from ray.rllib.agents import a3c, ppo
from ray.tune.logger import pretty_print

from rl_reader.callbacks.render_text import RenderTextCallback
from rl_reader.envs.sentpiece_mlm_super import SentPieceMLM


def main(args):
    ray.init()

    resources = ray.available_resources()
    num_gpus_available = resources.get('GPU', 0)
    num_gpus_driver = num_gpus_available * args.driver_gpu_ratio
    if args.num_procs:
        num_gpus_worker = (num_gpus_available - num_gpus_driver) / args.num_procs
        cpus_per_worker = (1 + args.num_procs) / resources['CPU']
    else:
        num_gpus_worker = 0
        cpus_per_worker = 0

    config = {}

    env_config = {
        'nlp_model_name': args.model_name,
        'corpus_path': args.corpus_path,
        'text_col': args.text_col,
        'base_reward': args.base_reward,
    }

    config.update({
        'framework': 'torch',
        'num_workers': args.num_procs,
        # 'num_cpus_for_driver': cpus_per_worker,
        # 'num_cpus_per_worker': cpus_per_worker,
        'num_envs_per_worker': args.num_envs_per_worker,
        'num_gpus': num_gpus_driver,
        'num_gpus_per_worker': num_gpus_worker,
        'train_batch_size': args.train_batch_size,
        'env_config': env_config,
        'callbacks': RenderTextCallback,
        'env': SentPieceMLM,
    })
    if 'sgd_minibatch_size' in config:
        config['sgd_minibatch_size'] = args.sgd_minibatch_size

    stop = dict(
        timesteps_total=args.total_steps
    )
    results = tune.run(
        args.agent, config=config, stop=stop,
        verbose=args.verbose, resume=args.resume,
        local_dir=args.results_dir
    )

    ray.shutdown()


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
    p.add_argument('--driver_gpu_ratio', default=0.2, type=float)
    p.add_argument('--num_envs_per_worker', default=1, type=int)
    p.add_argument('--train_batch_size', default=4000, type=int)
    p.add_argument('--sgd_minibatch_size', default=128, type=int)
    p.add_argument('--agent', default='PPO')
    p.add_argument('--base_reward', default=-0.1, type=float)
    p.add_argument('--verbose', default=2, type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--results_dir', default=None)
    args = p.parse_args()

    main(args)
