import torch

from rvsd.configs.baseline import RVSDBaselineConfig
from rvsd.tools.infer import run_inference


@torch.no_grad()
def run_evaluation(config: RVSDBaselineConfig, split: str = "test") -> dict[str, float]:
    print("[eval] deprecated alias: use run.mode=infer run.compute_metrics=true")
    result = run_inference(config, return_predictions=False, split=split, compute_metrics=True)
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    return metrics
