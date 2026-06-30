import numpy as np
import torch


SODA_HARD_THRESHOLD = 102.0 / 255.0
SODA_MAE_THRESHOLD = 12.0 / 255.0
SODA_BETA_SQUARE = 0.3
SODA_EPS = 1e-4


def compute_shadow_metrics(logits: torch.Tensor, mask_clip: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    targets = mask_clip > 0.5

    intersection = (preds & targets).sum().float()
    union = (preds | targets).sum().float().clamp_min(1.0)
    pred_area = preds.sum().float()
    target_area = targets.sum().float()
    dice = (2.0 * intersection) / (pred_area + target_area).clamp_min(1.0)

    return {
        "iou": float((intersection / union).item()),
        "dice": float(dice.item()),
    }


def _prediction_to_uint8(logit: torch.Tensor) -> np.ndarray:
    probability = torch.sigmoid(logit.detach()).squeeze().cpu().numpy()
    probability = np.clip(probability, 0.0, 1.0)
    return np.round(probability * 255.0).astype(np.uint8)


def _target_to_uint8(mask: torch.Tensor) -> np.ndarray:
    target = (mask.detach().squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return target * 255


def compute_soda_frame_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    hard_threshold: float = SODA_HARD_THRESHOLD,
    mae_threshold: float = SODA_MAE_THRESHOLD,
    use_soft_mae: bool = False,
) -> dict[str, np.ndarray | float]:
    assert prediction.dtype == np.uint8
    assert target.dtype == np.uint8
    assert prediction.shape == target.shape

    prediction_float = prediction.astype(np.float32) / 255.0
    target_float = target.astype(np.float32) / 255.0

    hard_target = (target_float > 0.5).astype(np.float32)
    positive_count = float(np.sum(hard_target))
    negative_count = float(hard_target.size - positive_count)

    if use_soft_mae:
        mae = float(np.mean(np.abs(prediction_float - target_float)))
    else:
        mae_prediction = (prediction_float > float(mae_threshold)).astype(np.float32)
        mae = float(np.mean(np.abs(mae_prediction - hard_target)))

    precision = np.zeros(256, dtype=np.float64)
    recall = np.zeros(256, dtype=np.float64)
    for curve_threshold in range(256):
        hard_prediction = (prediction > curve_threshold).astype(np.float32)
        true_positive = float(np.sum(hard_prediction * hard_target))
        predicted_positive = float(np.sum(hard_prediction))
        precision[curve_threshold] = (true_positive + SODA_EPS) / (predicted_positive + SODA_EPS)
        recall[curve_threshold] = (true_positive + SODA_EPS) / (positive_count + SODA_EPS)

    hard_threshold_u8 = float(np.clip(hard_threshold, 0.0, 1.0) * 255.0)
    prediction_binary = prediction.astype(np.float32) > hard_threshold_u8
    target_binary = hard_target.astype(bool)
    union = float(np.logical_or(prediction_binary, target_binary).sum())
    intersection = float(np.logical_and(prediction_binary, target_binary).sum())
    iou = intersection / max(union, 1.0)

    prediction_tmp = prediction_binary.astype(np.float32)
    target_tmp = target_binary.astype(np.float32)
    true_positive = float(np.sum(prediction_tmp * target_tmp))
    true_negative = float(np.sum((1.0 - prediction_tmp) * (1.0 - target_tmp)))
    positive_pixels = max(positive_count, 1.0)
    negative_pixels = max(negative_count, 1.0)
    shadow_accuracy = true_positive / positive_pixels
    non_shadow_accuracy = true_negative / negative_pixels
    ber = 0.5 * (2.0 - shadow_accuracy - non_shadow_accuracy) * 100.0
    shadow_ber = (1.0 - shadow_accuracy) * 100.0
    non_shadow_ber = (1.0 - non_shadow_accuracy) * 100.0

    return {
        "precision": precision,
        "recall": recall,
        "mae": mae,
        "iou": iou,
        "jaccard": iou,
        "ber": ber,
        "shadow_ber": shadow_ber,
        "non_shadow_ber": non_shadow_ber,
        "s_ber": shadow_ber,
        "n_ber": non_shadow_ber,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "positive_pixels": positive_count,
        "negative_pixels": negative_count,
    }


class SODAMetricAggregator:
    """
    Frame-level video shadow detection metrics aligned with SCOTCH/SODA testing.

    The CVPR 2023 SCOTCH/SODA public test code computes max F-beta from the
    averaged 256-threshold precision/recall curve, frame-averages IoU and MAE,
    and computes BER from dataset-level TP/TN/P/N sums. Its final test script
    uses a 102/255 hard threshold for BER/IoU and a 12/255 binary threshold for
    MAE; those are the defaults here.
    """

    def __init__(
        self,
        threshold: float | None = None,
        hard_threshold: float | None = None,
        mae_threshold: float = SODA_MAE_THRESHOLD,
        use_soft_mae: bool = False,
    ) -> None:
        if hard_threshold is None:
            hard_threshold = SODA_HARD_THRESHOLD if threshold is None else threshold
        self.hard_threshold = float(hard_threshold)
        self.mae_threshold = float(mae_threshold)
        self.use_soft_mae = bool(use_soft_mae)
        self.precision_sum = np.zeros(256, dtype=np.float64)
        self.recall_sum = np.zeros(256, dtype=np.float64)
        self.mae_sum = 0.0
        self.iou_sum = 0.0
        self.true_positive_sum = 0.0
        self.true_negative_sum = 0.0
        self.positive_pixel_sum = 0.0
        self.negative_pixel_sum = 0.0
        self.count = 0

    def update(self, logit: torch.Tensor, target: torch.Tensor) -> None:
        prediction_uint8 = _prediction_to_uint8(logit)
        target_uint8 = _target_to_uint8(target)
        frame_metrics = compute_soda_frame_metrics(
            prediction_uint8,
            target_uint8,
            hard_threshold=self.hard_threshold,
            mae_threshold=self.mae_threshold,
            use_soft_mae=self.use_soft_mae,
        )
        self.precision_sum += frame_metrics["precision"]
        self.recall_sum += frame_metrics["recall"]
        self.mae_sum += float(frame_metrics["mae"])
        self.iou_sum += float(frame_metrics["iou"])
        self.true_positive_sum += float(frame_metrics["true_positive"])
        self.true_negative_sum += float(frame_metrics["true_negative"])
        self.positive_pixel_sum += float(frame_metrics["positive_pixels"])
        self.negative_pixel_sum += float(frame_metrics["negative_pixels"])
        self.count += 1

    def summarize(self) -> dict[str, float]:
        if self.count == 0:
            return {
                "fmeasure": 0.0,
                "mae": 0.0,
                "iou": 0.0,
                "jaccard": 0.0,
                "ber": 0.0,
                "shadow_ber": 0.0,
                "non_shadow_ber": 0.0,
                "s_ber": 0.0,
                "n_ber": 0.0,
            }

        avg_precision = self.precision_sum / self.count
        avg_recall = self.recall_sum / self.count
        fmeasure_curve = (1.0 + SODA_BETA_SQUARE) * avg_precision * avg_recall / (
            SODA_BETA_SQUARE * avg_precision + avg_recall + 1e-12
        )
        iou = float(self.iou_sum / self.count)
        shadow_accuracy = self.true_positive_sum / max(self.positive_pixel_sum, 1.0)
        non_shadow_accuracy = self.true_negative_sum / max(self.negative_pixel_sum, 1.0)
        shadow_ber = float((1.0 - shadow_accuracy) * 100.0)
        non_shadow_ber = float((1.0 - non_shadow_accuracy) * 100.0)
        ber = float(0.5 * (2.0 - shadow_accuracy - non_shadow_accuracy) * 100.0)
        return {
            "fmeasure": float(np.max(fmeasure_curve)),
            "mae": float(self.mae_sum / self.count),
            "iou": iou,
            "jaccard": iou,
            "ber": ber,
            "shadow_ber": shadow_ber,
            "non_shadow_ber": non_shadow_ber,
            "s_ber": shadow_ber,
            "n_ber": non_shadow_ber,
        }


def build_soda_metric_aggregator(config_or_run=None) -> SODAMetricAggregator:
    run_config = getattr(config_or_run, "run", config_or_run)
    return SODAMetricAggregator(
        hard_threshold=float(getattr(run_config, "metric_hard_threshold", SODA_HARD_THRESHOLD)),
        mae_threshold=float(getattr(run_config, "metric_mae_threshold", SODA_MAE_THRESHOLD)),
        use_soft_mae=bool(getattr(run_config, "metric_use_soft_mae", False)),
    )
