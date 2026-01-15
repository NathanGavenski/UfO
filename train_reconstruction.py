from functools import partial
import resource
import argparse
from datetime import datetime
import os

from models.unsupervised import Unsupervised, CONFIG_FILE
from tensorboard_wrapper import Tensorboard

import gymnasium
import optuna
import numpy as np
from imitation_datasets.dataset import BaselineDataset
import torch
from torch.utils.data import DataLoader
from benchmark.methods.utils import import_hyperparameters
from tqdm import tqdm


def get_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()

    args.add_argument(
        "-g", "--game",
        type=str,
        default="Hopper-v4"
    )
    args.add_argument(
        "--optuna",
        action="store_true"
    )
    args.add_argument(
        "--att",
        action="store_true"
    )
    args.add_argument(
        "-w", "--weights",
        type=str,
        default=None,
    )
    args.add_argument(
        "--size",
        type=int,
        default=700
    )
    args.add_argument(
        "--epochs",
        type=int,
        default=1500000
    )

    return args.parse_args()


def objective(trial: optuna.Trial, args: argparse.Namespace):
    env = gymnasium.make(args.game)

    hyp = import_hyperparameters(
        CONFIG_FILE,
        env.spec.id,
    )
    if args.optuna:
        hyp["generator_lr"] = trial.suggest_float("generator_lr", 1e-5, 5e-3)
        hyp["lr"] = trial.suggest_float("lr", 1e-5, 5e-3)

    state_dataset = BaselineDataset(
        f"NathanGavenski/{args.game}",
        "HuggingFace",
        split="train",
        n_episodes=args.size
    )
    state_dataloader = DataLoader(state_dataset, 1024, shuffle=True)

    eval_dataset = BaselineDataset(
        f"NathanGavenski/{args.game}",
        "HuggingFace",
        split="eval",
        n_episodes=args.size
    )
    eval_dataloader = DataLoader(eval_dataset, 1024, shuffle=True)

    model = Unsupervised(
        env,
        enjoy_criteria=1000,
        hyperparameters=hyp,
        att=args.att,
    )
    if args.weights is not None:
        print(f"Loading from {args.weights}")
        model.load(args.weights)

    folder = f"./benchmark_results/reconstruction_stage/{args.game}/{args.size}"
    if not os.path.exists(folder):
        os.makedirs(f"{folder}/")

    name = datetime.now().strftime("%d/%m%Y-%H:%M:%S") if not args.debug else "test"
    if args.optuna:
        name = f"generative {hyp['generator_lr']} policy {hyp['lr']}"
    board = Tensorboard(path=folder, name=name, delete=args.debug)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.policy.to(device)
    model.generator.to(device)
    model.discriminator.to(device)

    best_model = -np.inf
    for epoch in tqdm(range(args.epochs if not args.optuna else 25)):
        model.freeze_model(model.generator)

        first_step_acc, \
            first_step_error, \
            first_step_complete, \
            first_step_homogenity = model._first_step(state_dataloader, mask=model.discrete)

        metrics = {
            "acc": first_step_acc,
            "error": first_step_error,
            "completeness": first_step_complete,
            "homogenity": first_step_homogenity,
            "lr": model.generator_optimizer.param_groups[0]['lr']
        }
        board.add_scalars("Train/First", epoch="train", **metrics)

        model.unfreeze_model(model.generator)
        model.freeze_model(model.policy)

        trajectories, lengths = model.run_episodes(100, env.spec.id)
        second_step_gen_error = model._second_step_gen(trajectories, lengths)

        metrics = {
            "error": second_step_gen_error,
        }
        board.add_scalars("Train/Second", epoch="train", **metrics)
        board.step("train")

        model.unfreeze_model(model.policy)

        with torch.no_grad():
            eval_first_step_acc, \
                eval_first_step_error, \
                eval_first_step_complete, \
                eval_first_step_homogenity = model._first_step(eval_dataloader, eval=True)

            metrics = {
                "acc": eval_first_step_acc,
                "error": eval_first_step_error,
                "completeness": eval_first_step_complete,
                "homogenity": eval_first_step_homogenity,
            }
            board.add_scalars("Eval", epoch="eval", **metrics)
            board.step("eval")

        if not args.optuna and epoch % model.enjoy_criteria == 0:
            model.policy.eval()
            aer = []
            for _ in tqdm(range(100), desc="eval"):
                obs, _ = env.reset()
                done = False
                acc_reward = 0

                while not done:
                    with torch.no_grad():
                        action = model.predict(obs)
                    obs, reward, done, terminated, info = env.step(action)
                    acc_reward += reward
                    done |= terminated
                aer.append(acc_reward)
            board.add_scalars("aer", "aer", aer=np.mean(aer), std=np.std(aer))
            board.step("aer")
            model.policy.train()

            if np.mean(aer) > best_model:
                print()
                print(f"Better model found: {np.mean(aer)}")
                print()
                best_model = np.mean(aer)
                path = "./unsupervised/reconstruction_stage"
                if args.optuna:
                    path += f"_att{args.att}_size{args.size}"

                if args.weights is not None:
                    path += "/finetune"

                path += f"/{args.game}/"
                model.save(path=path)

    return eval_first_step_error


if __name__ == "__main__":
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Setting limit to RLIMIT_NOFILE to: {hard}")
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except:
        print("no acess to set limit")

    args = get_args()

    if args.optuna:
        db = f"sqlite:///ufo-{args.game}.db"
        study = optuna.create_study(
            study_name="UfO",
            direction="minimize",
            storage=db,
            load_if_exists=True
        )

        objective = partial(objective, args=args)
        study.optimize(objective, n_trials=100)
    else:
        objective(None, args)
