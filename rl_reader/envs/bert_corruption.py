from copy import deepcopy as copy

import gym
from gym import spaces
import numpy as np
import torch


class BertCorruptionFinder(gym.Env):
    """Learn to identify corrupt tokens."""
    actions = [
        'move_left',
        'move_right',
        'toggle_corrupt',
        'finished',
    ]

    def __init__(self, docs, tokenizer, vec_model,
                 corruption_rate=0.2, reward_exploration=False,
                 base_reward=-0.1, informative_reward=True):
        self.docs = docs
        self.tokenizer = tokenizer
        self.vec_model = vec_model
        self._doc_vector_cache = {}
        self.corruption_rate = corruption_rate
        self.reward_exploration = reward_exploration
        self.base_reward = base_reward
        self.informative_reward = informative_reward

        sample_vecs = self.vectorize_doc(self.docs[0])
        self.action_space = spaces.Discrete(len(self.actions))
        # dimensionality of token embedding + 1 (is_corrupt)
        obs_dims = sample_vecs[0].shape[0] + 1
        print('observation dims', obs_dims)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * obs_dims),
            high=np.array([np.inf] * obs_dims),
        )
        # self.observation_space = (
        #     spaces.Discrete(sample_vecs[0].shape[0]),  # dimensionality of token embedding
        #     spaces.Discrete(1),  # is corrupt?
        # )

        self.reset()

    def reset(self):
        # corrupt the original tokens
        rand_doc_i = np.random.randint(len(self.docs))
        doc = self.docs[rand_doc_i]
        tokens = self.tokenizer(doc, return_tensors='pt', truncation=True)
        # corrupt the tokens
        num_tokens = len(tokens.input_ids[0])
        num_corrupted = int(num_tokens * self.corruption_rate)
        corrupted_indicies = np.random.choice(
            num_tokens, num_corrupted, replace=False
        ).astype(np.int32)
        self.is_corrupt = np.zeros((num_tokens, 1)).astype(np.bool)
        self.is_corrupt[corrupted_indicies] = True
        # get a random sample of the vocab and update the input tokens
        vocab_sample = self.sample_vocab(num_corrupted)
        tokens.input_ids[0][corrupted_indicies] = torch.from_numpy(vocab_sample)
        vectorized = self.vec_model(**tokens)[0][0].detach().cpu().numpy()
        self.tokens = vectorized
        # reset env state
        self.marked_corrupt = np.zeros((len(vectorized), 1)).astype(np.bool)
        self.visited = np.zeros((len(vectorized), 1)).astype(np.bool)
        self.cursor = 0

        return self.current_obs()

    def vectorize_doc(self, doc):
        tokens = self.tokenizer(doc, return_tensors='pt', truncation=True)
        vecs = self.vec_model(**tokens)[0][0]
        return vecs

    def sample_vocab(self, n):
        vocab = np.array(list(self.tokenizer.vocab.values()))
        indices = np.random.choice(vocab.shape[0], n, replace=False)
        return vocab[indices]

    def step(self, action_id):
        action_reward, is_done = self.dispatch_action(action_id)
        reward = float(self.base_reward + action_reward)
        obs = self.current_obs()
        return obs, reward, is_done, {}

    def current_obs(self):
        state = np.concatenate([
            self.tokens[self.cursor], self.is_corrupt[self.cursor]
        ], axis=-1)
        return state

    def dispatch_action(self, action_id):
        name = self.actions[action_id]
        f_action = getattr(self, name)
        return f_action()

    def move_left(self):
        self.cursor = max(0, self.cursor - 1)
        return 0., False

    def move_right(self):
        self.cursor = min(len(self.tokens) - 1, self.cursor + 1)
        return 0., False

    def toggle_corrupt(self):
        self.marked_corrupt[self.cursor] = not self.marked_corrupt[self.cursor]
        if self.informative_reward:
            reward = self.marked_corrupt[self.cursor] == self.is_corrupt[self.cursor]
            reward = (reward - 1) * 3. + 1.  # [-2, 1] lose more than gain for incorrect toggle
            reward = float(reward)
        else:
            reward = 0.
        return reward, False

    def finished(self):
        reward = (self.marked_corrupt == self.is_corrupt).sum() / sum(self.is_corrupt)
        return reward, True
