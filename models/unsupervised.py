'''Module for Unsupervised Behavioural Cloning from Observation'''
from datetime import datetime
from collections import defaultdict
from torch.multiprocessing import Pool
from typing import Any, Dict, List, Tuple
import os

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import gymnasium as gym
from gymnasium import Env
from tensorboard_wrapper.tensorboard import Tensorboard
from imitation_datasets.dataset.metrics import accuracy as accuracy_fn
from benchmark.methods.policies.mlp import MlpWithAttention
from benchmark.methods.method import Metrics, Method
from benchmark.methods.utils import import_hyperparameters

from utils import get_clusters, get_clustering_metrics
from utils import error as error_fn
from models.mlp import Policy
from models.utils.discriminator import DiscLSTM
from custom_datasets.agent_dataset import StateDataset, TrajectoryDataset, DiscriminatorDataset

CONFIG_FILE = "config/unsupervised.yaml"


class Unsupervised(Method):
    def __init__(
        self,
        environment: Env,
        enjoy_criteria: int = 100,
        verbose: bool = False,
        config_file: str = None,
        hyperparameters: Dict[str, Any] = None,
        att: bool = True,
    ) -> None:
        self.enjoy_criteria = enjoy_criteria
        self.verbose = verbose
        self.environment_name = environment.spec.name
        self.save_path = f"./tmp/unsupervised/{self.environment_name}/"

        if config_file is None:
            config_file = CONFIG_FILE

        self.hyperparameters = import_hyperparameters(
            config_file,
            environment.spec.id,
        )
        if hyperparameters is not None:
            for key, value in hyperparameters.items():
                self.hyperparameters[key] = value

        super().__init__(
            environment,
            self.hyperparameters,
            continuous_loss=nn.L1Loss
        )
        self.save_path = f"unsupervised/{self.environment_name}/"
        self.is_training = False

        policy = self.hyperparameters.get("policy", "MlpPolicy")
        if policy == "MlpPolicy":
            print("using custom policy")
            activation_fn = nn.LeakyReLU if self.discrete else nn.Tanh
            self.policy = Policy(
                self.observation_size,
                self.action_size,
                activation=activation_fn,
                att=att,
            )

        generator = self.hyperparameters.get('generator', 'MlpPolicy')
        if generator == 'MlpPolicy':
            activation_fn = nn.LeakyReLU if self.discrete else nn.Tanh
            self.generator = Policy(
                self.observation_size + self.action_size,
                self.observation_size,
                activation=activation_fn,
                att=att,
            )
        elif generator == 'MlpWithAttention':
            self.generator = MlpWithAttention(
                self.observation_size + self.action_size,
                self.observation_size
            )
        self.generator.to(self.device)

        parameters = list(self.generator.parameters()) + list(self.policy.parameters())
        self.optimizer_fn = optim.Adam(
            parameters,
            lr=self.hyperparameters['lr'],
            weight_decay=self.hyperparameters.get('wd_lr', 0)
        )
        self.generator_optimizer = optim.Adam(
            parameters,
            lr=self.hyperparameters['generator_lr'],
            weight_decay=self.hyperparameters.get('wd_gen', 0)
        )
        self.adversarial_optimizer = optim.Adam(
            parameters,
            lr=self.hyperparameters['adversarial_lr'],
            weight_decay=self.hyperparameters.get('wd_adv', 0)
        )
        self.generator_loss = nn.L1Loss()

        discriminator = self.hyperparameters.get("discriminator", "MlpPolicy")
        if discriminator == "RNNPolicy":
            self.discriminator = DiscLSTM(
                input_dim=environment.observation_space.shape[0],
                hidden_dim=64,
                num_layer=2,
                output_dim=2,
                device=self.device
            )
        else:
            raise AttributeError("Only RNN Discriminator implemented")

        self.discriminator.to(self.device)
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.hyperparameters['discriminator_lr'],
            weight_decay=self.hyperparameters.get('wd_dic', 0)
        )
        self.discriminator_loss = nn.CrossEntropyLoss()

    def freeze_model(self, model: torch.nn.Module) -> None:
        for params in model.parameters():
            params.requires_grad = False

    def unfreeze_model(self, model: torch.nn.Module) -> None:
        for params in model.parameters():
            params.requires_grad = True

    def forward(self, x: torch.Tensor, clip: bool = True) -> torch.Tensor:
        x = self.policy(x)
        if clip:
            x = torch.clamp(x, min=-1, max=1)
        return x

    def policy_predict(
        self,
        x: torch.Tensor,
        apply_mask: bool = False,
        clip: bool = False
    ) -> torch.Tensor:
        x = self.forward(x, clip)
        if apply_mask:
            x = F.softmax(x, dim=-1)
            _, max_indices = torch.max(x, dim=-1, keepdim=True)
            x_hard = torch.zeros_like(x).scatter_(-1, max_indices, 1.0)
            x = (x_hard - x).detach() + x
        return x

    def save_hparams(self, board: Tensorboard) -> None:
        board.add_hparams(hparams=self.hyperparameters)

    def save(self, path: str = None, name: str = None) -> None:
        path = self.save_path if path is None else path
        if not os.path.exists(path):
            os.makedirs(path)

        policy_name = "best_model" if name is None else f"best_model_{name}"
        generator_name = "generator" if name is None else f"generator_{name}"
        discriminator_name = "discriminator" if name is None else f"discriminator_{name}"
        torch.save(self.policy.state_dict(), f"{path}/{policy_name}.ckpt")
        torch.save(self.generator.state_dict(), f"{path}/{generator_name}.ckpt")
        torch.save(self.discriminator.state_dict(), f"{path}/{discriminator_name}.ckpt")

    def load(self, path: str = None) -> Self:
        path = self.save_path if path is None else path

        if not os.path.exists(path):
            raise ValueError("Path does not exists.")

        self.policy.load_state_dict(
            torch.load(
                f"{path}best_model.ckpt",
                map_location=torch.device(self.device)
            )
        )
        self.generator.load_state_dict(
            torch.load(
                f"{path}generator.ckpt",
                map_location=torch.device(self.device)
            )
        )

        try:
            self.discriminator.load_state_dict(
                torch.load(
                    f"{path}discriminator.ckpt",
                    map_location=torch.device(self.device)
                )
            )
        except RuntimeError:
            pass

        return self

    def get_dradient_norm(self, model: nn.Module) -> list[float]:
        grad_norm = []
        for _, param in model.named_parameters():
            if param.grad is not None:
                grad_norm.append(param.grad.norm().item())
        return grad_norm

    def train(
        self,
        n_epochs: int,
        train_dataset: Dict[str, DataLoader],
        eval_dataset: Dict[str, DataLoader] = None,
        folder: str = None,
        board_name: str = None,
    ) -> Self:
        self.is_training = True

        if folder is None:
            folder = f"./benchmark_results/unsupervised/{self.environment_name}"

        if not os.path.exists(folder):
            os.makedirs(f"{folder}/")
        name = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
        if board_name is not None:
            name = board_name
        self.save_path = f"{self.save_path}{name}/"
        board = Tensorboard(path=folder, name=name)
        self.save_hparams(board)
        self.policy.to(self.device)
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        best_model = -np.inf
        pbar = range(n_epochs)
        pbar = tqdm(pbar, desc="Main train")
        for epoch in pbar:
            train_metrics = self._train(**train_dataset)
            for prior, metrics in train_metrics.items():
                board.add_scalars(prior, epoch="train", **metrics)

            if epoch % self.enjoy_criteria == 0 and epoch > 0:
                with torch.no_grad():
                    eval_metrics = self._eval()
                    board.add_scalars("eval", epoch="eval", **eval_metrics)

                    if eval_metrics['aer'] > best_model:
                        self.save()
                        best_model = eval_metrics['aer']

            board.step()

        self.is_training = False
        return self

    def run_episode(self, env_name: str) -> Tuple[torch.Tensor, int]:
        with gym.make(env_name) as environment:
            state, _ = environment.reset()
            states = [state]
            done = False
            count = 0
            while not done:
                print(count, done)
                action = self.predict(state)
                state, reward, done, truncated, info = environment.step(action)
                done |= truncated
                states.append(state)
                count += 1
        return torch.as_tensor(states), len(states)

    def run_with_progress_bar(self, pool, function, arguments, total):
        results = []
        with tqdm(total=total, desc="Collecting Samples") as pbar:
            def update_bar(*_):
                pbar.update()

            # Submit tasks
            for arg in arguments:
                result = pool.apply_async(function, args=(arg,), callback=update_bar)
                results.append(result)

            # Gather results
            output = [res.get() for res in results]
        return output

    def run_episodes1(self, num_episodes, env_name):
        with Pool(processes=4) as pool:
            arguments = [(env_name) for _ in range(num_episodes)]
            results = pool.starmap(self.run_episode, zip(arguments))
        trajectories, lengths = zip(*results)
        return list(trajectories), list(lengths)

    def run_episodes(self, num_episodes, env_name):
        trajectories = []
        lengths = []

        pbar = range(num_episodes)
        if self.verbose:
            pbar = tqdm(range(num_episodes), desc="Collecting Samples")

        for episode in pbar:
            with gym.make(env_name) as environment:
                state, _ = environment.reset()
                done = False
                trajectory = [state]
                while not done:
                    action = self.predict(state)
                    state, reward, done, truncated, info = environment.step(action)
                    done |= truncated
                    trajectory.append(state)
            trajectories.append(torch.as_tensor(trajectory))
            lengths.append(len(trajectory))
        return trajectories, lengths

    def _train(
        self,
        expert_dataset: DataLoader,
        signature_dataset: DataLoader[DiscriminatorDataset]
    ) -> Metrics:
        if not self.policy.training:
            self.policy.train()

        if not self.generator.training:
            self.generator.train()

        # First step of training (policy + generative model)
        first_step_acc, \
            first_step_error, \
            first_step_complete, \
            first_step_homogenity = self._first_step(expert_dataset)

        # Collect data
        times = 100
        trajectories, lengths = self.run_episodes(times, self.environment.spec.id)
        signature_dataset.dataset.append_agent_trajectories(trajectories, lengths)
        signature_dataset.dataset.set_dataset_trajectories()

        # Second step (policy + generative + discrminator)
        second_step_gen_error = self._second_step_gen(trajectories, lengths)
        second_step_acc, second_step_loss = self._second_step_disc(trajectories, lengths)

        # Third step (discriminator)
        third_step_acc, third_step_loss = self._third_step_disc(signature_dataset)

        return {
            "first_step": {
                "acc": first_step_acc,
                "error": first_step_error,
                "completeness": first_step_complete,
                "homogenity": first_step_homogenity,
            },
            "second_step": {
                "acc": second_step_acc,
                "error_adversarial": second_step_loss,
                "error_generative": second_step_gen_error,
            },
            "third_step": {
                "acc": third_step_acc,
                "error": third_step_loss,
            }
        }

    def _first_step(
        self,
        expert_dataset: DataLoader,
        eval: bool = False,
        mask: bool = False
    ) -> Metrics:
        error = []
        acc = []
        predicted = defaultdict(list)
        pbar = tqdm(expert_dataset, desc="First Step") if self.verbose else expert_dataset
        for batch in pbar:
            state, action, next_state = batch
            state = state.to(self.device)
            next_state = next_state.to(self.device)

            self.generator_optimizer.zero_grad()
            pred_action = self.policy_predict(
                state,
                apply_mask=self.discrete,
                clip=not self.discrete
            )
            if self.discrete:
                acc.append(accuracy_fn(pred_action, action.squeeze(1)))
                get_clusters(pred_action, action, predicted, True)
            else:
                acc.append(error_fn(pred_action, action))

            G_input = torch.cat((state, pred_action), dim=1)
            pred_next_state = self.generator(G_input)
            loss = self.generator_loss(next_state, pred_next_state)

            if not eval:
                loss.backward()
                self.generator_optimizer.step()

            error.append(loss.item())
        acc = np.mean(acc)
        error = np.mean(error)
        completness, homogenity = 0, 0
        if self.discrete:
            completness, homogenity = get_clustering_metrics(predicted, self.action_size)

        return acc, error, completness, homogenity

    def _second_step_gen(
        self,
        trajectories: List[torch.Tensor],
        lengths: List[int]
    ) -> Metrics:
        state_dataset = DataLoader(
            StateDataset(trajectories, lengths),
            128,
            shuffle=True
        )
        error = []
        pbar = state_dataset
        if self.verbose:
            pbar = tqdm(state_dataset, desc="second step (gen)")

        for batch in pbar:
            states, next_states = batch
            states = states.to(self.device)
            next_states = next_states.to(self.device)

            self.generator_optimizer.zero_grad()
            pred_action = self.policy_predict(
                states,
                apply_mask=self.discrete,
                clip=not self.discrete
            )

            G_input = torch.cat((states, pred_action), dim=1)

            pred_next_state = self.generator(G_input)

            loss = self.generator_loss(next_states, pred_next_state)
            loss.backward()
            self.generator_optimizer.step()

            error.append(loss.item())
        error = np.mean(error)
        return error

    def _second_step_disc(
        self,
        trajectories: List[torch.Tensor],
        lengths: List[int],
    ) -> Metrics:
        trajectory_dataset = DataLoader(
            TrajectoryDataset(trajectories, lengths),
            16,
            shuffle=True
        )
        acc = []
        error = []
        pbar = tqdm(trajectory_dataset,
                    desc="Second Step (DISC)") if self.verbose else trajectory_dataset
        for batch in pbar:
            trajectories, length = batch
            trajectories = trajectories.to(self.device)

            B, L, C = trajectories.shape
            reshaped_trajectories = trajectories.reshape(-1, C)

            self.adversarial_optimizer.zero_grad()
            actions = self.policy_predict(reshaped_trajectories)
            G_input = torch.cat((reshaped_trajectories, actions), dim=1)

            G_trajectories = self.generator(G_input)
            G_trajectories = G_trajectories.reshape(B, L, C)

            final_trajectories = torch.cat((
                trajectories[:, 0, :].reshape(B, 1, C),
                G_trajectories[:, :-1, :]
            ), dim=1)
            delta = final_trajectories[1:] - final_trajectories[:-1]

            predictions = self.discriminator(delta, lengths)
            predictions = predictions.reshape(-1, 2)

            loss = self.discriminator_loss(
                predictions,
                torch.ones(predictions.size(0)).long().to(self.device)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.adversarial_optimizer.step()

            Y = torch.zeros(predictions.size(0)).long()
            acc.append(accuracy_fn(predictions, Y))
            error.append(loss.item())
        acc = np.mean(acc)
        error = np.mean(error)
        return acc, error

    def _third_step_disc(self, signature_dataset: DataLoader) -> Metrics:
        acc = []
        error = []
        pbar = signature_dataset
        if self.verbose:
            pbar = tqdm(signature_dataset, desc="Third Step")

        for trajectories, lenght, Y in pbar:
            trajectories = trajectories.to(self.device)
            lenght = lenght.to(self.device)
            Y = Y.to(self.device)

            self.discriminator_optimizer.zero_grad()
            predictions = self.discriminator(trajectories, lenght)
            B, S, C, = predictions.size()
            predictions = predictions.reshape(-1, 2)
            Y = Y.view(-1, C)
            Y = Y.repeat(1, S)
            Y = Y.view(-1)

            loss = self.discriminator_loss(predictions, Y)
            loss.backward()
            self.discriminator_optimizer.step()

            error.append(loss.item())
            acc.append(accuracy_fn(predictions, Y))
        acc = np.mean(acc)
        error = np.mean(error)
        return acc, error

    def _eval(self) -> Metrics:
        rewards = []
        for i in range(100):
            env = self.environment
            state, _ = env.reset(seed=i)
            done = False
            accumulated_reward = 0
            while not done:
                action = self.predict(state)
                state, reward, done, truncated, info = env.step(action)
                done |= truncated
                accumulated_reward += reward
            rewards.append(accumulated_reward)

        return {'aer': np.mean(rewards), 'std': np.std(rewards)}
