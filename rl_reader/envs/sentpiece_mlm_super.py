from copy import deepcopy as copy

import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from .render_text import render_text


class SentPieceMLM(gym.Env):
    """Learn to fill in masked tokens with a single agent."""
    cursor_actions = [
        'move_left',
        'move_right',
        'set_token',
        'finished',
    ]

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, env_config):
        self.setup(env_config)
        self.corruption_rate = env_config.get('corruption_rate', 0.2)
        self.reward_exploration = env_config.get('reward_exploration', False)
        self.base_reward = env_config.get('base_reward', -0.1)
        self.informative_reward = env_config.get('informative_reward', True)

        self.mask_token_id = self.tokenizer.vocab[self.tokenizer.mask_token]
        self.vocab_size = len(self.tokenizer.vocab)
        obs_dims = self.vocab_size
        print('observation dims', obs_dims)
        self.observation_space = spaces.Discrete(obs_dims)

        self.cursor_action_space = spaces.Discrete(len(self.cursor_actions))
        self.token_action_space = spaces.Discrete(self.vocab_size)
        self.action_space = spaces.MultiDiscrete([
            len(self.cursor_actions), self.vocab_size
        ])

        self.reset()

    def setup(self, conf):
        self.docs = self.load_docs(
            conf['corpus_path'], text_col=conf.get('text_col', 'text')
        )
        self.tokenizer = self.load_tokenizer(
            conf.get('hf_model_name', 'bert-base-uncased')
        )

    def reset(self):
        # corrupt the original tokens
        rand_doc_i = np.random.randint(len(self.docs))
        doc = self.docs[rand_doc_i]
        token_info = self.tokenizer(
            doc, truncation=True, return_offsets_mapping=True
        )
        self.token_offset_mapping = token_info.offset_mapping
        tokens = np.array(token_info.input_ids)
        self.original_tokens = copy(tokens)
        # corrupt the tokens
        num_tokens = len(tokens)
        num_corrupted = max(int(num_tokens * self.corruption_rate), 1)
        corrupted_indicies = np.random.choice(
            num_tokens, num_corrupted, replace=False
        ).astype(np.int32)
        self.is_corrupt = np.zeros((num_tokens,)).astype(np.bool)
        self.is_corrupt[corrupted_indicies] = True

        tokens[corrupted_indicies] = self.mask_token_id
        self.tokens = tokens
        # reset env state
        self.marked_corrupt = np.zeros((len(tokens),)).astype(np.bool)
        self.visited = np.zeros((len(tokens),)).astype(np.bool)
        self.cursor = 0

        self.num_cursor_steps = 0
        self.token_agent_id = "token_{}".format(
            self.num_cursor_steps
        )

        return self.current_obs()

    def current_obs(self):
        return self.tokens[self.cursor]

    def step(self, actions):
        cursor_action, token_action = actions
        self.num_cursor_steps += 1

        obs, reward, is_done, info = self._step(cursor_action, token_action)
        reward = reward + self.base_reward
        return obs, reward, is_done, info

    def _step(self, action, token):
        name = self.cursor_actions[action]
        f_action = getattr(self, name)
        obs, reward, is_done, info = f_action(token)
        return obs, reward, is_done, info

    def set_token(self, new_token_id):
        self.marked_corrupt[self.cursor] = True
        reward = 0.
        if self.informative_reward:
            old_token_id = self.tokens[self.cursor]
            token_was_masked = self.is_corrupt[self.cursor]
            if old_token_id == new_token_id:
                reward = -1.
            else:
                reward += token_was_masked
                new_token_is_correct = \
                    new_token_id == self.original_tokens[self.cursor]
                reward += new_token_is_correct

        self.tokens[self.cursor] = new_token_id
        obs = self.current_obs()
        is_done = False

        return obs, reward, is_done, {}

    def move_left(self, *args):
        self.cursor = max(0, self.cursor - 1)
        return self.current_obs(), self.base_reward, False, {}

    def move_right(self, *args):
        self.cursor = min(len(self.tokens) - 1, self.cursor + 1)
        return self.current_obs(), self.base_reward, False, {}

    def finished(self, *args):
        num_corrupt = self.is_corrupt.sum()
        if num_corrupt == 0:
            num_corrupt = 1
        reward = -(self.tokens != self.original_tokens).sum() / num_corrupt
        # print(self.render())
        return self.current_obs(), reward + self.base_reward, True, {}

    def render(self, *args, **kwargs):
        decoded = self.tokenizer.convert_ids_to_tokens(self.tokens)
        # decoded = self.tokenizer.decode(self.tokens)
        correct = self.tokens == self.original_tokens
        return render_text(
            decoded, correct, self.marked_corrupt, self.is_corrupt,
        )
        # return render_text(
        #     decoded, correct, self.marked_corrupt, self.is_corrupt,
        #     self.token_offset_mapping
        # )
        # return ' '.join(decoded)

    def load_docs(self, path, text_col='text'):
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_pickle(path)
        text = df[text_col]
        return text

    def load_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
