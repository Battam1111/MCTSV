import random
from collections import deque, namedtuple
import numpy as np
import torch

# 定义一个简单的命名元组来存储经验
Experience = namedtuple('Experience', 
                        ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity, batch_size, device='cpu'):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device

    def push(self, state, action, reward, next_state, done):
        state = np.array(state, np.float32)
        next_state = np.array(next_state, np.float32)
        action = np.array([action], np.int64)
        reward = np.array([reward], np.float32)
        done = np.array([done], np.float32)

        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.buffer, self.batch_size)

        states, actions, rewards, next_states, dones = map(np.stack, zip(*experiences))
        return (
            torch.tensor(states, device=self.device, dtype=torch.float),
            torch.tensor(actions, device=self.device, dtype=torch.long).squeeze(-1),
            torch.tensor(rewards, device=self.device, dtype=torch.float).squeeze(-1),
            torch.tensor(next_states, device=self.device, dtype=torch.float),
            torch.tensor(dones, device=self.device, dtype=torch.float).squeeze(-1)
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, device='cpu'):
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.device = device
        self.experiences = deque(maxlen=capacity)
        self.priorities = SegmentTree(capacity)
        self.capacity = capacity
        self.position = 0

    def push(self, experience):
        max_priority = max(self.priorities.tree[self.capacity:self.capacity+self.priorities.size]) if self.priorities.size > 0 else 1.0
        if len(self.experiences) < self.capacity:
            self.experiences.append(experience)
        else:
            self.experiences[self.position] = experience
        self.priorities.update(self.position, max_priority ** self.alpha)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_indices = []
        p_total = self.priorities.sum(0, self.capacity)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx = self.priorities.find_prefixsum_idx(s)
            batch_indices.append(idx)

        sampled_experiences = [self.experiences[idx] for idx in batch_indices]

        # 将每个元素解压缩为单独的列表
        states, actions, rewards, next_states, dones = zip(*sampled_experiences)

        # 分别处理全局信息流和局部空间矩阵
        states_global = torch.stack([torch.tensor(s[0], dtype=torch.float32, device=self.device) for s in states])
        states_local = torch.stack([torch.tensor(s[1], dtype=torch.float32, device=self.device) for s in states])
        actions = torch.tensor(actions, device=self.device, dtype=torch.long).squeeze(-1)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float).squeeze(-1)
        next_states_global = torch.stack([torch.tensor(s[0], dtype=torch.float32, device=self.device) for s in next_states])
        next_states_local = torch.stack([torch.tensor(s[1], dtype=torch.float32, device=self.device) for s in next_states])
        dones = torch.tensor(dones, device=self.device, dtype=torch.float).squeeze(-1)

        weights = [(self.priorities.tree[self.capacity + idx] / p_total) ** (-self.beta) for idx in batch_indices]
        weights = torch.tensor(weights, device=self.device, dtype=torch.float)
        weights /= weights.max()  # 归一化以处理权重

        return (
            (states_global, states_local),
            actions,
            rewards,
            (next_states_global, next_states_local),
            dones,
            weights,
            torch.tensor(batch_indices, device=self.device, dtype=torch.long)
        )


    def update_priorities(self, indices, priorities):
        assert np.all(priorities > 0), "Priorities must be positive."
        for idx, priority in zip(indices, priorities):
            assert 0 <= idx < len(self.experiences), "Index out of bounds."
            self.priorities.update(idx, priority.item() ** self.alpha)


        
    def __len__(self):
        return len(self.experiences)

    
class SegmentTree:
    def __init__(self, capacity):
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2  # 保证capacity为2的幂，简化实现
        self.tree = np.zeros(2 * self.capacity, dtype=np.float32)
        self.size = 0

    def _update(self, idx, val):
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]
            idx //= 2

    def update(self, idx, val):
        self._update(idx, val)
        self.size = min(self.size + 1, self.capacity)

    def sum(self, left, right):
        result = 0.0
        left += self.capacity
        right += self.capacity
        while left < right:
            if left % 2 == 1:
                result += self.tree[left]
                left += 1
            if right % 2 == 1:
                right -= 1
                result += self.tree[right]
            left //= 2
            right //= 2
        return result

    def find_prefixsum_idx(self, s):
        idx = 1
        while idx < self.capacity:
            if self.tree[2 * idx] > s:
                idx = 2 * idx
            else:
                s -= self.tree[2 * idx]
                idx = 2 * idx + 1
        return idx - self.capacity

