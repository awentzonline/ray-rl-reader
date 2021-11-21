import multiprocessing

import gym
import ray
from ray.rllib.agents import a3c, ppo
from ray.tune.logger import pretty_print

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

    agent_module = dict(
        ppo=ppo, a2c=a3c
    )[args.agent]
    config = agent_module.DEFAULT_CONFIG.copy()

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
        'train_batch_size': args.train_batch_size,
        'env_config': env_config,
    })
    if 'sgd_minibatch_size' in config:
        config['sgd_minibatch_size'] = args.sgd_minibatch_size

    trainer_cls = dict(
        ppo=ppo.PPOTrainer,
        a2c=a3c.A2CTrainer,
        a3c=a3c.A3CTrainer,
    )[args.agent]
    trainer = trainer_cls(
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
    p.add_argument('--train_batch_size', default=4000, type=int)
    p.add_argument('--sgd_minibatch_size', default=128, type=int)
    p.add_argument('--agent', default='ppo')
    args = p.parse_args()

    main(args)
