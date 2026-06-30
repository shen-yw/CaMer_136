import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from rvsd.configs.baseline import RVSDBaselineConfig


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class ShadowFreeDataset(Dataset):
    """
    Training-only auxiliary dataset for clips or images that contain no shadows.

    Returns:
    - clip: [3, T, H, W]
    - mask_clip: [T, 1, H, W] filled with zeros
    - meta: source metadata plus a shadow_free flag
    """

    def __init__(
        self,
        dataset_root: str,
        split_name: str,
        image_dir_name: str,
        clip_length: int,
        stride: int,
        image_size: tuple[int, int],
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
        self.clip_length = clip_length
        self.stride = stride
        self.image_size = image_size
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

    def _resolve_image_root(self) -> Path:
        candidates = []
        if self.split_name:
            split_root = self.dataset_root / self.split_name
            candidates.extend(
                [
                    split_root / self.image_dir_name,
                    split_root,
                ]
            )
        candidates.extend(
            [
                self.dataset_root / self.image_dir_name,
                self.dataset_root,
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not resolve a shadow-free image root from dataset_root={self.dataset_root} "
            f"split_name={self.split_name!r} image_dir_name={self.image_dir_name!r}"
        )

    def _build_dummy_samples(self) -> list[dict[str, object]]:
        return [
            {
                "video_id": f"shadow_free_dummy_{sample_idx:04d}",
                "frame_paths": [Path(f"frame_{frame_idx:05d}.jpg") for frame_idx in range(self.clip_length)],
                "frame_names": [f"frame_{frame_idx:05d}" for frame_idx in range(self.clip_length)],
                "is_padding": [False for _ in range(self.clip_length)],
                "start_index": 0,
                "source_kind": "dummy",
            }
            for sample_idx in range(self.dummy_num_samples)
        ]

    def _list_video_dirs(self, root: Path) -> list[Path]:
        return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name)

    def _frame_files_by_stem(self, root: Path) -> dict[str, Path]:
        files = sorted(
            [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES],
            key=lambda path: path.name,
        )
        return {path.stem: path for path in files}

    def _select_video_names_for_split(self, video_names: list[str]) -> set[str]:
        selected = list(video_names)
        if 0.0 < self.train_video_split_ratio < 1.0 and len(selected) > 1:
            shuffled_names = list(selected)
            rng = random.Random(self.train_video_split_seed)
            rng.shuffle(shuffled_names)
            train_count = int(round(len(shuffled_names) * self.train_video_split_ratio))
            train_count = min(max(train_count, 1), len(shuffled_names) - 1)
            selected = shuffled_names[:train_count] if self.training else shuffled_names[train_count:]
        if self.max_videos is not None:
            selected = selected[: self.max_videos]
        return set(selected)

    def _build_video_samples(self, image_root: Path) -> list[dict[str, object]]:
        video_dirs = self._list_video_dirs(image_root)
        if not video_dirs:
            return []
        selected_names = self._select_video_names_for_split([path.name for path in video_dirs])
        samples: list[dict[str, object]] = []
        for video_dir in video_dirs:
            if video_dir.name not in selected_names:
                continue
            frame_files = self._frame_files_by_stem(video_dir)
            if not frame_files:
                continue
            ordered_stems = sorted(frame_files.keys())
            ordered_paths = [frame_files[stem] for stem in ordered_stems]
            clip_count = 0
            for start_index in range(0, len(ordered_paths), self.stride):
                clip_paths = list(ordered_paths[start_index : start_index + self.clip_length])
                clip_names = list(ordered_stems[start_index : start_index + self.clip_length])
                is_padding = [False for _ in range(len(clip_paths))]
                while len(clip_paths) < self.clip_length:
                    clip_paths.append(clip_paths[-1])
                    clip_names.append(clip_names[-1])
                    is_padding.append(True)
                samples.append(
                    {
                        "video_id": video_dir.name,
                        "frame_paths": clip_paths,
                        "frame_names": clip_names,
                        "is_padding": is_padding,
                        "start_index": start_index,
                        "source_kind": "video",
                    }
                )
                clip_count += 1
                if self.max_clips_per_video is not None and clip_count >= self.max_clips_per_video:
                    break
        return samples

    def _build_image_samples(self, image_root: Path) -> list[dict[str, object]]:
        frame_files = self._frame_files_by_stem(image_root)
        if not frame_files:
            return []
        ordered_items = sorted(frame_files.items(), key=lambda item: item[0])
        if self.max_videos is not None:
            ordered_items = ordered_items[: self.max_videos]
        samples = []
        for stem, path in ordered_items:
            samples.append(
                {
                    "video_id": stem,
                    "frame_paths": [path for _ in range(self.clip_length)],
                    "frame_names": [stem for _ in range(self.clip_length)],
                    "is_padding": [frame_index > 0 for frame_index in range(self.clip_length)],
                    "start_index": 0,
                    "source_kind": "image",
                }
            )
        return samples

    def _build_real_samples(self) -> list[dict[str, object]]:
        image_root = self._resolve_image_root()
        samples = self._build_video_samples(image_root)
        if not samples:
            samples = self._build_image_samples(image_root)
        if not samples:
            raise RuntimeError(f"No shadow-free clips/images indexed from {image_root}")
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

    def _make_dummy_clip(self, index: int) -> torch.Tensor:
        height, width = self.image_size
        generator = torch.Generator().manual_seed(index)
        return torch.rand(3, self.clip_length, height, width, generator=generator)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        if self.dummy_mode:
            clip = self._make_dummy_clip(index)
            original_sizes = [self.image_size for _ in range(self.clip_length)]
        else:
            params = self._sample_transform_params()
            image_tensors = []
            original_sizes = []
            for frame_path in sample["frame_paths"]:
                image = Image.open(frame_path).convert("RGB")
                original_sizes.append((image.height, image.width))
                image_tensors.append(self._apply_image_transform(image, params))
            clip = torch.stack(image_tensors, dim=1)

        height, width = self.image_size
        mask_clip = torch.zeros(self.clip_length, 1, height, width, dtype=clip.dtype)
        return {
            "clip": clip,
            "mask_clip": mask_clip,
            "meta": {
                "video_id": sample["video_id"],
                "start_index": sample["start_index"],
                "frame_names": list(sample["frame_names"]),
                "image_paths": [str(path) for path in sample["frame_paths"]],
                "label_paths": ["<implicit_zero_mask>" for _ in sample["frame_paths"]],
                "is_padding": list(sample["is_padding"]),
                "original_sizes": original_sizes,
                "split_name": self.split_name,
                "shadow_free": True,
                "source_kind": sample["source_kind"],
            },
        }


def collate_shadow_free_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "clip": torch.stack([sample["clip"] for sample in batch], dim=0),
        "mask_clip": torch.stack([sample["mask_clip"] for sample in batch], dim=0),
        "meta": [sample["meta"] for sample in batch],
    }


def build_shadow_free_dataloader(
    config: RVSDBaselineConfig,
    training: bool = True,
    shuffle: bool | None = None,
) -> DataLoader:
    aux_cfg = config.shadow_free_aux
    if not aux_cfg.dataset_root:
        raise ValueError("shadow_free_aux.dataset_root must be set before building the auxiliary dataloader.")
    if shuffle is None:
        shuffle = training

    dataset = ShadowFreeDataset(
        dataset_root=aux_cfg.dataset_root,
        split_name=aux_cfg.split_name,
        image_dir_name=aux_cfg.image_dir_name,
        clip_length=aux_cfg.clip_length,
        stride=aux_cfg.stride,
        image_size=aux_cfg.image_size,
        training=training,
        dummy_mode=aux_cfg.dummy_mode,
        dummy_num_samples=aux_cfg.dummy_num_samples,
        max_videos=aux_cfg.max_videos,
        max_clips_per_video=aux_cfg.max_clips_per_video,
        train_video_split_ratio=aux_cfg.train_video_split_ratio,
        train_video_split_seed=aux_cfg.train_video_split_seed,
        hflip_prob=aux_cfg.hflip_prob,
        color_jitter_prob=aux_cfg.color_jitter_prob,
        color_jitter_brightness=aux_cfg.color_jitter_brightness,
        color_jitter_contrast=aux_cfg.color_jitter_contrast,
        color_jitter_saturation=aux_cfg.color_jitter_saturation,
        color_jitter_hue=aux_cfg.color_jitter_hue,
    )

    batch_size = aux_cfg.batch_size if aux_cfg.batch_size > 0 else config.run.batch_size
    num_workers = aux_cfg.num_workers if aux_cfg.num_workers >= 0 else config.run.num_workers
    prefetch_factor = aux_cfg.prefetch_factor if aux_cfg.prefetch_factor > 0 else config.run.prefetch_factor
    if aux_cfg.persistent_workers is None:
        persistent_workers = bool(getattr(config.run, "persistent_workers", True))
    else:
        persistent_workers = bool(aux_cfg.persistent_workers)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0 and persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_shadow_free_batch,
    )
