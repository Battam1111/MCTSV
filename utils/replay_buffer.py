import random
from collections import deque, namedtuple
import numpy as np
import torch

# 分别定义两个命名元组来存储经验
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


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
        # Assign the maximum priority for new experience
        max_priority = self.priorities.max() if self.priorities.size > 0 else 1.0
        if len(self.experiences) < self.capacity:
            self.experiences.append(experience)
        else:
            self.experiences[self.position] = experience
        self.priorities.update(self.position, max_priority ** self.alpha)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        p_total = self.priorities.sum(0, self.capacity)
        segment = p_total / batch_size
        batch_indices = [self.priorities.find_prefixsum_idx(random.uniform(segment * i, segment * (i + 1)))
                         for i in range(batch_size)]

        experiences = [self.experiences[idx] for idx in batch_indices]

        # 处理全局和局部状态，它们尺寸不同
        states_global = torch.stack([torch.tensor(exp[0][0], dtype=torch.float32, device=self.device) for exp in experiences])
        states_local = torch.stack([torch.tensor(exp[0][1], dtype=torch.float32, device=self.device) for exp in experiences])
        actions = torch.tensor([exp[1] for exp in experiences], dtype=torch.long, device=self.device).squeeze(-1)
        rewards = torch.tensor([exp[2] for exp in experiences], dtype=torch.float, device=self.device).squeeze(-1)
        next_states_global = torch.stack([torch.tensor(exp[3][0], dtype=torch.float32, device=self.device) for exp in experiences])
        next_states_local = torch.stack([torch.tensor(exp[3][1], dtype=torch.float32, device=self.device) for exp in experiences])
        dones = torch.tensor([exp[4] for exp in experiences], dtype=torch.float, device=self.device).squeeze(-1)

        # 计算权重
        weights = []
        max_weight = float('-inf')
        for idx in batch_indices:
            priority = self.priorities.tree[self.capacity + idx]
            weight = (priority / p_total) ** (-self.beta)
            weights.append(weight)
            max_weight = max(max_weight, weight)

        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        weights /= max_weight  # 归一化权重

        return ((states_global, states_local), actions, rewards, (next_states_global, next_states_local), dones, weights, torch.tensor(batch_indices, device=self.device, dtype=torch.long))

    def update_priorities(self, indices, priorities):
        # 确保所有优先级都为正数
        priorities = np.array([max(priority, 1e-6) for priority in priorities])
        assert np.all(priorities > 0), "Priorities must be positive."
        for idx, priority in zip(indices, priorities):
            assert 0 <= idx < len(self.experiences), "Index out of bounds."
            self.priorities.update(idx, priority ** self.alpha)


    def __len__(self):
        return len(self.experiences)

class SegmentTree:
    def __init__(self, capacity):
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
        # 初始化时设定一个小的正数，确保没有零值
        self.tree = np.full(2 * self.capacity, 1e-6, dtype=np.float32)
        self.size = 0
        self.max_weight = 1e-6  # Track the maximum weight for normalization

    def update(self, idx, val):
        idx += self.capacity
        # 确保新的优先级不为零
        val = max(val, 1e-6)
        self.tree[idx] = val
        # 更新最大权重
        self.max_weight = max(self.max_weight, val)
        while idx > 1:
            idx //= 2
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def sum(self, left, right):
        # Calculate the sum over a range
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
        # Find the highest index with a prefix sum greater than s
        idx = 1
        while idx < self.capacity:
            if self.tree[2 * idx] > s:
                idx = 2 * idx
            else:
                s -= self.tree[2 * idx]
                idx = 2 * idx + 1
        return idx - self.capacity

    def max(self):
        # Return the maximum priority
        return self.max_weight
