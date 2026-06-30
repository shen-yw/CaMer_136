import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as TF

from rvsd.configs.baseline import RVSDBaselineConfig


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _resolve_split_name(config: RVSDBaselineConfig, split: str) -> str:
    if split == "train":
        return config.dataset.train_split_name
    if split == "val":
        return config.dataset.train_split_name
    if split == "test":
        return config.dataset.test_split_name
    return split


class VideoShadowDataset(Dataset):
    """
    Returns:
    - clip: [3, T, H, W]
    - mask_clip: [T, 1, H, W]
    - meta: video id, frame names, original sizes, paths, and padding flags
    """

    def __init__(
        self,
        dataset_root: str,
        split_name: str,
        image_dir_name: str,
        label_dir_name: str,
        clip_length: int,
        stride: int,
        image_size: tuple[int, int],
        logical_split: str = "train",
        training: bool = False,
        dummy_mode: bool = False,
        dummy_num_samples: int = 0,
        max_videos: int | None = None,
        max_clips_per_video: int | None = None,
        train_video_split_ratio: float = 0.8,
        train_video_split_seed: int = 42,
        hflip_prob: float = 0.5,
        color_jitter_prob: float = 0.8,
        color_jitter_brightness: float = 0.1,
        color_jitter_contrast: float = 0.1,
        color_jitter_saturation: float = 0.05,
        color_jitter_hue: float = 0.02,
    ):
        if clip_length < 1:
            raise ValueError(f"clip_length must be >= 1, got {clip_length}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.dataset_root = Path(dataset_root).expanduser()
        self.split_name = split_name
        self.image_dir_name = image_dir_name
        self.label_dir_name = label_dir_name
        self.clip_length = clip_length
        self.stride = stride
        self.image_size = image_size
        self.logical_split = logical_split
        self.training = training
        self.dummy_mode = dummy_mode
        self.dummy_num_samples = dummy_num_samples
        self.max_videos = max_videos
        self.max_clips_per_video = max_clips_per_video
        self.train_video_split_ratio = train_video_split_ratio
        self.train_video_split_seed = train_video_split_seed
        self.hflip_prob = hflip_prob
        self.color_jitter_prob = color_jitter_prob
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_saturation = color_jitter_saturation
        self.color_jitter_hue = color_jitter_hue

        if self.dummy_mode:
            self.samples = self._build_dummy_samples()
        else:
            self.samples = self._build_real_samples()

    def _build_dummy_samples(self) -> list[dict[str, object]]:
        return [
            {
                "video_id": f"dummy_{self.split_name}_{sample_idx:04d}",
                "frame_pairs": [
                    {
                        "image_path": Path(f"frame_{frame_idx:05d}.jpg"),
                        "label_path": Path(f"frame_{frame_idx:05d}.png"),
                        "stem": f"frame_{frame_idx:05d}",
                        "is_padding": False,
                    }
                    for frame_idx in range(self.clip_length)
                ],
                "start_index": 0,
            }
            for sample_idx in range(self.dummy_num_samples)
        ]

    def _split_root(self) -> Path:
        return self.dataset_root / self.split_name

    def _image_root(self) -> Path:
        return self._split_root() / self.image_dir_name

    def _label_root(self) -> Path:
        return self._split_root() / self.label_dir_name

    def _list_video_dirs(self, root: Path) -> list[Path]:
        return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name)

    def _select_video_names_for_split(self, video_names: list[str]) -> set[str]:
        if self.logical_split not in {"train", "val"}:
            selected = video_names
        else:
            if not 0.0 < self.train_video_split_ratio < 1.0:
                raise ValueError(
                    f"train_video_split_ratio must be in (0, 1), got {self.train_video_split_ratio}"
                )
            shuffled_names = list(video_names)
            rng = random.Random(self.train_video_split_seed)
            rng.shuffle(shuffled_names)
            if len(shuffled_names) <= 1:
                selected = shuffled_names if self.logical_split == "train" else []
            else:
                train_count = int(round(len(shuffled_names) * self.train_video_split_ratio))
                train_count = min(max(train_count, 1), len(shuffled_names) - 1)
                if self.logical_split == "train":
                    selected = shuffled_names[:train_count]
                else:
                    selected = shuffled_names[train_count:]
        if self.max_videos is not None:
            selected = selected[: self.max_videos]
        return set(selected)

    def _frame_files_by_stem(self, root: Path) -> dict[str, Path]:
        files = sorted(
            [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES],
            key=lambda path: path.name,
        )
        return {path.stem: path for path in files}

    def _build_real_samples(self) -> list[dict[str, object]]:
        split_root = self._split_root()
        image_root = self._image_root()
        label_root = self._label_root()

        if not split_root.exists():
            raise FileNotFoundError(f"Missing split directory: {split_root}")
        if not image_root.exists():
            raise FileNotFoundError(f"Missing image directory: {image_root}")
        if not label_root.exists():
            raise FileNotFoundError(f"Missing label directory: {label_root}")

        image_video_dirs = self._list_video_dirs(image_root)
        label_video_dirs = self._list_video_dirs(label_root)
        if not image_video_dirs:
            raise RuntimeError(f"No video folders found under {image_root}")

        image_video_names = {path.name for path in image_video_dirs}
        label_video_names = {path.name for path in label_video_dirs}
        if image_video_names != label_video_names:
            missing_in_labels = sorted(image_video_names - label_video_names)
            missing_in_images = sorted(label_video_names - image_video_names)
            raise RuntimeError(
                f"Image/label video folder mismatch. "
                f"Missing in labels: {missing_in_labels[:5]} Missing in images: {missing_in_images[:5]}"
            )

        selected_video_names = self._select_video_names_for_split(sorted(image_video_names))
        image_video_dirs = [path for path in image_video_dirs if path.name in selected_video_names]
        if not image_video_dirs:
            raise RuntimeError(f"No video folders selected for logical split '{self.logical_split}' under {image_root}")

        samples: list[dict[str, object]] = []
        for image_video_dir in image_video_dirs:
            video_id = image_video_dir.name
            label_video_dir = label_root / video_id
            image_frames = self._frame_files_by_stem(image_video_dir)
            label_frames = self._frame_files_by_stem(label_video_dir)

            if not image_frames:
                raise RuntimeError(f"No image frames found in {image_video_dir}")
            if not label_frames:
                raise RuntimeError(f"No label frames found in {label_video_dir}")

            image_stems = set(image_frames.keys())
            label_stems = set(label_frames.keys())
            if image_stems != label_stems:
                missing_labels = sorted(image_stems - label_stems)
                missing_images = sorted(label_stems - image_stems)
                raise RuntimeError(
                    f"Frame mismatch for video {video_id}. "
                    f"Missing labels: {missing_labels[:5]} Missing images: {missing_images[:5]}"
                )

            ordered_stems = sorted(image_frames.keys())
            frame_pairs = [
                {
                    "image_path": image_frames[stem],
                    "label_path": label_frames[stem],
                    "stem": stem,
                    "is_padding": False,
                }
                for stem in ordered_stems
            ]

            video_clip_count = 0
            for start_index in range(0, len(frame_pairs), self.stride):
                clip_pairs = [dict(pair) for pair in frame_pairs[start_index : start_index + self.clip_length]]
                while len(clip_pairs) < self.clip_length:
                    padded = dict(clip_pairs[-1])
                    padded["is_padding"] = True
                    clip_pairs.append(padded)
                samples.append(
                    {
                        "video_id": video_id,
                        "frame_pairs": clip_pairs,
                        "start_index": start_index,
                    }
                )
                video_clip_count += 1
                if self.max_clips_per_video is not None and video_clip_count >= self.max_clips_per_video:
                    break

        if not samples:
            raise RuntimeError(f"No clips indexed from {split_root}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_transform_params(self) -> dict[str, object]:
        do_hflip = self.training and random.random() < self.hflip_prob
        do_color_jitter = self.training and random.random() < self.color_jitter_prob
        return {
            "do_hflip": do_hflip,
            "do_color_jitter": do_color_jitter,
            "brightness_factor": 1.0 + random.uniform(-self.color_jitter_brightness, self.color_jitter_brightness),
            "contrast_factor": 1.0 + random.uniform(-self.color_jitter_contrast, self.color_jitter_contrast),
            "saturation_factor": 1.0 + random.uniform(-self.color_jitter_saturation, self.color_jitter_saturation),
            "hue_factor": random.uniform(-self.color_jitter_hue, self.color_jitter_hue),
        }

    def _resize_tensor(self, tensor: torch.Tensor, mode: str) -> torch.Tensor:
        kwargs = {"size": self.image_size, "mode": mode}
        if mode != "nearest":
            kwargs["align_corners"] = False
        return F.interpolate(tensor.unsqueeze(0), **kwargs).squeeze(0)

    def _apply_image_transform(self, image: Image.Image, params: dict[str, object]) -> torch.Tensor:
        tensor = TF.to_tensor(image)
        if params["do_hflip"]:
            tensor = TF.hflip(tensor)
        if params["do_color_jitter"]:
            tensor = TF.adjust_brightness(tensor, float(params["brightness_factor"]))
            tensor = TF.adjust_contrast(tensor, float(params["contrast_factor"]))
            tensor = TF.adjust_saturation(tensor, float(params["saturation_factor"]))
            tensor = TF.adjust_hue(tensor, float(params["hue_factor"]))
        return self._resize_tensor(tensor, mode="bilinear")

    def _apply_mask_transform(self, mask: Image.Image, params: dict[str, object]) -> torch.Tensor:
        array = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        tensor = torch.from_numpy(array).unsqueeze(0)
        if params["do_hflip"]:
            tensor = TF.hflip(tensor)
        return self._resize_tensor(tensor, mode="nearest")

    def _make_dummy_clip(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = self.image_size
        generator = torch.Generator().manual_seed(index)
        clip = torch.rand(3, self.clip_length, height, width, generator=generator)
        mask_clip = (torch.rand(self.clip_length, 1, height, width, generator=generator) > 0.5).float()
        return clip, mask_clip

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        if self.dummy_mode:
            clip, mask_clip = self._make_dummy_clip(index)
            original_sizes = [self.image_size for _ in range(self.clip_length)]
        else:
            params = self._sample_transform_params()
            image_tensors: list[torch.Tensor] = []
            label_tensors: list[torch.Tensor] = []
            original_sizes: list[tuple[int, int]] = []

            for pair in sample["frame_pairs"]:
                image = Image.open(pair["image_path"]).convert("RGB")
                mask = Image.open(pair["label_path"]).convert("L")
                original_sizes.append((image.height, image.width))
                image_tensors.append(self._apply_image_transform(image, params))
                label_tensors.append(self._apply_mask_transform(mask, params))

            clip = torch.stack(image_tensors, dim=1)
            mask_clip = torch.stack(label_tensors, dim=0)

        return {
            "clip": clip,
            "mask_clip": mask_clip,
            "meta": {
                "video_id": sample["video_id"],
                "start_index": sample["start_index"],
                "frame_names": [pair["stem"] for pair in sample["frame_pairs"]],
                "image_paths": [str(pair["image_path"]) for pair in sample["frame_pairs"]],
                "label_paths": [str(pair["label_path"]) for pair in sample["frame_pairs"]],
                "is_padding": [bool(pair["is_padding"]) for pair in sample["frame_pairs"]],
                "original_sizes": original_sizes,
                "split_name": self.split_name,
                "logical_split": self.logical_split,
            },
        }


def collate_video_shadow_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "clip": torch.stack([sample["clip"] for sample in batch], dim=0),
        "mask_clip": torch.stack([sample["mask_clip"] for sample in batch], dim=0),
        "meta": [sample["meta"] for sample in batch],
    }


def build_dataloader(
    config: RVSDBaselineConfig,
    split: str,
    training: bool | None = None,
    shuffle: bool | None = None,
    sampler: DistributedSampler | None = None,
) -> DataLoader:
    split_name = _resolve_split_name(config, split)
    if training is None:
        training = split == "train"
    if shuffle is None:
        shuffle = training

    dataset = VideoShadowDataset(
        dataset_root=config.dataset.dataset_root,
        split_name=split_name,
        image_dir_name=config.dataset.image_dir_name,
        label_dir_name=config.dataset.label_dir_name,
        clip_length=config.dataset.clip_length,
        stride=config.dataset.stride,
        image_size=config.dataset.image_size,
        logical_split=split,
        training=training,
        dummy_mode=config.dataset.dummy_mode,
        dummy_num_samples=config.dataset.dummy_num_samples,
        max_videos=config.dataset.max_videos,
        max_clips_per_video=config.dataset.max_clips_per_video,
        train_video_split_ratio=config.dataset.train_video_split_ratio,
        train_video_split_seed=config.dataset.train_video_split_seed,
        hflip_prob=config.dataset.hflip_prob,
        color_jitter_prob=config.dataset.color_jitter_prob,
        color_jitter_brightness=config.dataset.color_jitter_brightness,
        color_jitter_contrast=config.dataset.color_jitter_contrast,
        color_jitter_saturation=config.dataset.color_jitter_saturation,
        color_jitter_hue=config.dataset.color_jitter_hue,
    )
    return DataLoader(
        dataset,
        batch_size=config.run.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.run.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.run.num_workers > 0 and bool(getattr(config.run, "persistent_workers", True)),
        prefetch_factor=config.run.prefetch_factor if config.run.num_workers > 0 else None,
        collate_fn=collate_video_shadow_batch,
    )
