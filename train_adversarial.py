import argparse
from copy import deepcopy
from collections import defaultdict
from functools import partial
from datetime import datetime
import os
import resource
import shutil
from typing import Any, Union, Callable

from benchmark.methods.utils import import_hyperparameters
import gymnasium
from models.unsupervised import Unsupervised, CONFIG_FILE
import numpy as np
import optuna
from tensorboard_wrapper import Tensorboard
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_datasets.agent_dataset import DiscriminatorDataset


class ModelSaver:

    def __init__(
        self,
        save_fn: Callable,
        key: Union[str, list[str]],
        direction: Union[str, list[str]] = "maximize",
        quantity: int = 5,
    ):
        self.key = key
        self.save_fn = save_fn
        self.direction = direction
        self.quantity = quantity
        self.collection = defaultdict(lambda: defaultdict(float))

    def save_model(self, path: str, metrics: dict[str, Any]) -> None:
        if len(self.collection) < self.quantity:
            self.add(path, metrics)
        elif self.should_add(metrics):
            self.add(path, metrics)

    def should_add(self, metrics: dict[str, Any]) -> bool:
        for key, value in self.collection.items():
            if isinstance(self.key, list):
                raise NotImplementedError("multiple keys not supported yet")
            else:
                if self.direction == "maximize":
                    return metrics[self.key] >= value[self.key]
                if self.direction == "minimize":
                    return metrics[self.key] <= value[self.key]

    def find_index(self, metrics: [dict, Any]) -> int:
        index = -1
        for key, value in self.collection.items():
            if isinstance(self.key, list):
                raise NotImplementedError("multiple keys not supported yet")
            else:
                if self.direction == "maximize" and metrics[self.key] >= value[self.key]:
                    index = key
                if self.direction == "minimize" and metrics[self.key] <= value[self.key]:
                    index = key

        if len(self.collection) < self.quantity and index == -1:
            return len(self.collection)
        return index

    def add(self, path: str, metrics: dict[str, Any]) -> None:
        if len(self.collection) < self.quantity:
            index = len(self.collection)
            self.collection[index] = metrics
            self.save_fn(f"{path}/{str(index)}")
            print(f"Saving model with {metrics} in index {index}")
            self.order_indexes(path)
            return

        index = self.find_index(metrics)
        self.swap_indexes(index, path)
        self.collection[index] = metrics
        self.save_fn(f"{path}/{str(index)}")
        print(f"Saving model with {metrics} in index {index}")

    def swap_indexes(self, index, path) -> None:
        if index == self.quantity - 1:
            return

        indexes = list(self.collection.keys())[index:][::-1]
        for idx, next_idx in zip(indexes, indexes[1:]):
            self.collection[idx] = deepcopy(self.collection[next_idx])
            shutil.move(f"{path}/{next_idx}", f"{path}/{idx}")
            print(f"Moving {path}/{next_idx} to {path}/{idx}")

    def order_indexes(self, path):
        while True:
            updated = False

            for i in range(len(self.collection) - 1):
                st_metric = self.collection[i][self.key]
                nd_metric = self.collection[i + 1][self.key]
                if (self.direction == "maximize" and st_metric < nd_metric) or \
                        (self.direction == "minimize" and st_metric > nd_metric):
                    updated = True
                    shutil.move(f"{path}/{i}", f"{path}/tmp")
                    shutil.move(f"{path}/{i + 1}", f"{path}/{i}")
                    shutil.move(f"{path}/tmp", f"{path}/{i + 1}")
                    tmp = self.collection[i]
                    self.collection[i] = self.collection[i + 1]
                    self.collection[i + 1] = tmp
                    print(f"Moved {i} ({st_metric}) to {i + 1} ({nd_metric})")

            if not updated:
                break


def get_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()

    args.add_argument(
        "-g", "--game",
        type=str,
        default="Hopper-v4"
    )

    args.add_argument(
        "-w", "--weights",
        type=str,
        default="./unsupervised/reconstruction_stage/"
    )
    args.add_argument(
        "--optuna",
        action="store_true",
    )
    args.add_argument(
        "--debug",
        action="store_true",
    )
    args.add_argument(
        "--att",
        action="store_true",
    )

    return args.parse_args()


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    env = gymnasium.make(args.game)
    hyperparameters = import_hyperparameters(CONFIG_FILE, env.spec.id)
    if args.optuna:
        hyperparameters["adversarial_lr"] = trial.suggest_float("adversarial_lr", 1e-6, 5e-3)

    if args.optuna:
        hyperparameters["discriminator_lr"] = trial.suggest_float("discriminator_lr", 1e-6, 5e-3)

    # Setup model
    model = Unsupervised(
        env,
        enjoy_criteria=10,
        hyperparameters=hyperparameters,
        att=args.att
    )
    model.policy.load_state_dict(torch.load(f"{args.weights}best_model.ckpt"))
    model.generator.load_state_dict(torch.load(f"{args.weights}generator.ckpt"))

    saver = ModelSaver(model.save, "aer")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.policy.to(device)
    model.generator.to(device)
    model.discriminator.to(device)
    model.freeze_model(model.generator)

    # Setup Tensorboard
    folder = f"./benchmark_results/ail/{args.game}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    name = datetime.now().strftime("%d/%m/%Y-%H:%M:%S") if not args.debug else "test"
    if args.optuna:
        name = f"adv {hyperparameters['adversarial_lr']}/disc {hyperparameters['discriminator_lr']}"
    board = Tensorboard(path=folder, name=name, delete=args.debug)

    # Setup dataset
    discriminator_dataset = DiscriminatorDataset(
        f"NathanGavenski/{args.game}",
        "HuggingFace",
    )
    discriminator_dataloader = DataLoader(discriminator_dataset, 8, shuffle=True)

    # Train
    metrics = defaultdict(lambda: defaultdict(float))
    best_model = -np.inf
    for epoch in tqdm(range(10 if not args.optuna else 5)):
        trajectories, lenghts = model.run_episodes(100, env.spec.id)
        discriminator_dataloader.dataset.append_agent_trajectories(
            trajectories, lenghts
        )
        discriminator_dataloader.dataset.set_dataset_trajectories()

        # second step disc
        acc, error = model._second_step_disc(trajectories, lenghts)
        metrics["adversarial"]["loss"] = error
        metrics["adversarial"]["acc"] = acc
        board.add_scalars("Train/Adversarial", epoch="train", **metrics["adversarial"])

        # third step disc
        acc, error = model._third_step_disc(discriminator_dataloader)
        metrics["disc"]["loss"] = error
        metrics["disc"]["acc"] = acc
        board.add_scalars("Train/Discriminator", epoch="train", **metrics["disc"])
        board.step("train")

        with torch.no_grad():
            eval_metrics = model._eval()
            aer, std = eval_metrics["aer"], eval_metrics["std"]
            board.add_scalars("Eval", epoch="eval", aer=aer, std=std)
            board.step("eval")
            if aer > best_model:
                best_model = aer

        if not args.optuna:
            path = f"./unsupervised/ail/{args.game}"
            saver.save_model(path, eval_metrics)

    return aer


if __name__ == "__main__":
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Setting limit to RLIMIT_NOFILE to: {hard}")
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except:
        print("No acess to set limit")

    args = get_args()

    if args.optuna:
        db = f"sqlite:///discriminator-{args.game}.db"
        study = optuna.create_study(
            study_name="discriminator",
            direction="maximize",
            storage=db,
            load_if_exists=True
        )

        objective = partial(objective, args=args)
        study.optimize(objective, n_trials=100)
    else:
        objective(None, args)
