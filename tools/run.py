import os
import random
import sys

from omegaconf import OmegaConf
import numpy as np
import torch
import torch.multiprocessing as mp

from rvsd.configs.baseline import build_baseline_config
from rvsd.tools.eval import run_evaluation
from rvsd.tools.infer import run_inference
from rvsd.tools.speed import run_speed_test
from rvsd.tools.speed_real import run_real_speed_test
from rvsd.tools.train import run_smoke, train_model


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def cli_to_dict(argv: list[str]) -> dict[str, object]:
    return OmegaConf.to_container(OmegaConf.from_cli(argv), resolve=True) or {}


def _spawn_worker(local_rank: int, world_size: int, config) -> None:
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = config.run.distributed_master_addr
    os.environ["MASTER_PORT"] = str(config.run.distributed_master_port)
    seed_everything(int(config.run.seed) + int(local_rank))
    train_model(config, overfit_one_batch=(config.run.mode == "overfit"))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    config = build_baseline_config(cli_to_dict(argv))
    mode = config.run.mode

    if not (
        mode in {"train", "overfit"}
        and config.run.distributed_world_size > 1
        and "LOCAL_RANK" not in os.environ
    ):
        seed_everything(int(config.run.seed))

    if (
        mode in {"train", "overfit"}
        and config.run.distributed_world_size > 1
        and "LOCAL_RANK" not in os.environ
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("distributed_world_size > 1 requires CUDA.")
        if torch.cuda.device_count() < config.run.distributed_world_size:
            raise RuntimeError(
                f"Requested distributed_world_size={config.run.distributed_world_size}, "
                f"but only found {torch.cuda.device_count()} CUDA devices."
            )
        mp.spawn(
            _spawn_worker,
            args=(config.run.distributed_world_size, config),
            nprocs=config.run.distributed_world_size,
            join=True,
        )
        return 0

    if mode == "smoke":
        run_smoke(config, split=config.run.split_name)
    elif mode == "train":
        train_model(config, overfit_one_batch=False)
    elif mode == "overfit":
        train_model(config, overfit_one_batch=True)
    elif mode == "eval":
        run_evaluation(config)
    elif mode == "infer":
        run_inference(config, return_predictions=False, compute_metrics=config.run.compute_metrics)
    elif mode == "speed":
        run_speed_test(config, split=config.run.split_name)
    elif mode == "speed_real":
        run_real_speed_test(config, split=config.run.split_name)
    else:
        raise ValueError(f"Unsupported run mode: {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
