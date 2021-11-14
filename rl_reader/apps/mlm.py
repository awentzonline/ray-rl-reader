import multiprocessing

import gym
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

from rl_reader.envs.sentpiece_mlm import SentPieceMLM


def main(args):
    ray.init()

    resources = ray.available_resources()
    num_gpus_available = resources.get('GPU', 0)
    num_gpus_driver = num_gpus_available * args.driver_gpu_ratio
    num_gpus_worker = (num_gpus_available - num_gpus_driver) / args.num_procs
    cpus_per_worker = (1 + args.num_procs) / resources['CPU']

    config = ppo.DEFAULT_CONFIG.copy()

    def policy_mapping_fn(agent_id, *args, **kwargs):
        if agent_id.startswith('token_'):
            return 'token_policy'
        else:
            return 'cursor_policy'

    env_config = {
        'nlp_model_name': args.model_name,
        'corpus_path': args.corpus_path,
        'text_col': args.text_col,
    }
    env = SentPieceMLM(env_config)

    config.update({
        'framework': 'torch',
        'num_workers': args.num_procs,
        # 'num_cpus_for_driver': cpus_per_worker,
        # 'num_cpus_per_worker': cpus_per_worker,
        'num_envs_per_worker': args.num_envs_per_worker,
        'num_gpus': num_gpus_driver,
        'num_gpus_per_worker': num_gpus_worker,
        'multiagent': {
            'policies': {
                'cursor_policy': (
                    None, env.observation_space,
                    env.cursor_action_space, {
                        'model': {
                            'use_lstm': True,
                        },
                        'gamma': 0.9
                    }
                ),
                'token_policy': (
                    None, env.observation_space,
                    env.token_action_space, {
                        'model': {
                            'use_lstm': True,
                        },
                        'gamma': 0.9
                    }
                ),
            },
            'policy_mapping_fn': policy_mapping_fn,
        },
        'env_config': env_config,
    })

    trainer = ppo.PPOTrainer(
        config=config, env=SentPieceMLM,
    )

    try:
        for i in range(1000):
           # Perform one iteration of training the policy with PPO
           result = trainer.train()
           print(pretty_print(result))
    except KeyboardInterrupt:
        pass

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
    args = p.parse_args()

    main(args)
