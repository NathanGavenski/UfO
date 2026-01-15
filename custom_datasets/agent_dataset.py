from __future__ import annotations

from copy import deepcopy

from imitation_datasets.dataset import BaselineDataset
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_sequence, pack_padded_sequence
from tqdm import tqdm


class StateDataset(Dataset):
    def __init__(self, trajectories, lengths):
        self.trajectories = trajectories
        self.lengths = lengths

        states = []
        next_states = []
        for trajectory in trajectories:
            for state, next_state in zip(trajectory[:-1], trajectory[1:]):
                states.append(state.tolist())
                next_states.append(next_state.tolist())

        self.states = torch.tensor(states)
        self.next_states = torch.tensor(next_states)

    def __len__(self):
        return self.states.size(0)

    def __getitem__(self, index):
        return self.states[index], self.next_states[index]


class TrajectoryDataset(Dataset):

    def __init__(self, trajectories, lengths):
        self.trajectories = pad_sequence(trajectories, batch_first=True)
        self.lengths = lengths

    def __len__(self):
        return self.trajectories.size(0)

    def __getitem__(self, index):
        return self.trajectories[index], self.lengths[index]


class DiscriminatorDataset(BaselineDataset):

    def __init__(
        self,
        path: str,
        source: str = "hf",
        split: str = "train",
        n_episodes: int = None,
    ):
        super().__init__(path, source, split, n_episodes)

        episode_starts = list(np.where(self.data["episode_starts"] == 1)[0])
        episode_starts.append(len(self.data["episode_starts"]))

        if n_episodes is not None:
            if split == "train":
                episode_starts = episode_starts[: n_episodes + 1]
            else:
                episode_starts = episode_starts[n_episodes:]

        # Expert trajectories
        trajectories, lengths = [], []
        for start, end in zip(
            episode_starts, tqdm(episode_starts[1:], desc="Creating sequence")
        ):
            episode = self.data["obs"][start:end]
            trajectories.append(torch.from_numpy(episode))
            lengths.append(end - start)
        self.expert_trajectories = pad_sequence(trajectories, batch_first=True)
        self.expert_lengths = lengths

        self.buffer_max_size = self.expert_trajectories.size(0)
        self.max_size = 0
        self.agent_max_size = self.expert_trajectories.size(1)

        # Agent trajectories
        self.agent_trajectories = torch.Tensor(
            size=(0, *self.expert_trajectories.shape[1:])
        )
        self.agent_lengths = []

        # Dataset trajectories
        self.trajectories = deepcopy(self.expert_trajectories)
        self.lengths = deepcopy(self.expert_lengths)

    def get_subtrajectories(
        self,
        trajectories: list[torch.Tensor[float]],
        lenghts: list[int],
    ) -> tuple[torch.Tensor[float], list[int]]:
        """Get a set of trajectories and lengths of these trajectories and build
        all subtrajectories."""
        subtrajectories, trajectory_lenghts = [], []
        for trajectory in trajectories:
            for idx in range(1, len(trajectory) + 1):
                if isinstance(trajectory, np.ndarray):
                    trajectory = torch.from_numpy(trajectory)
                subtrajectories.append(trajectory[:idx])
                trajectory_lenghts.append(idx)
        subtrajectories = pad_sequence(subtrajectories, batch_first=True)
        return subtrajectories, trajectory_lenghts

    def append_agent_trajectories(
        self,
        trajectories: list[torch.Tensor[float]],
        lenghts: list[int],
    ) -> None:
        trajectories_tmp = [torch.tensor(trajectory) for trajectory in trajectories]
        trajectories = pad_sequence(trajectories_tmp, batch_first=True)
        if trajectories.size(1) < self.agent_max_size:
            pad_size = self.agent_max_size - trajectories.size(1)
            trajectories = torch.nn.functional.pad(trajectories, (0, 0, 0, pad_size), value=0)
        self.agent_trajectories = torch.cat(
            (self.agent_trajectories, trajectories), dim=0
        )
        self.agent_lengths += lenghts
        self.max_size = self.agent_trajectories.size(0)
        if self.max_size > self.buffer_max_size:
            index = self.max_size - self.buffer_max_size
            self.agent_trajectories = self.agent_trajectories[index:]
            self.max_size = self.agent_trajectories.size(0)

    def set_dataset_trajectories(self):
        indexes = np.random.randint(
            0, self.agent_trajectories.size(0), self.max_size
        ).astype(int)
        tmp_agent_trajectories = self.agent_trajectories[indexes]
        tmp_agent_lengths = np.array(self.agent_lengths)[indexes]

        indexes = np.random.randint(
            0, self.expert_trajectories.size(0), self.max_size
        ).astype(int)
        tmp_expert_trajectories = self.expert_trajectories[indexes]
        tmp_expert_lengths = np.array(self.expert_lengths)[indexes]

        self.trajectories = torch.cat(
            (tmp_expert_trajectories, tmp_agent_trajectories), dim=0
        )

        self.lengths = np.append(tmp_expert_lengths, tmp_agent_lengths, axis=0)
        assert self.trajectories.size(0) == len(self.lengths)

    def __len__(self):
        return self.trajectories.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor[float], torch.Tensor[int]]:
        trajectory = self.trajectories[index]
        delta = trajectory[1:] - trajectory[:-1]
        length = self.lengths[index]
        y = torch.tensor(1) if index < self.max_size else torch.tensor(0)
        return delta, length, y
