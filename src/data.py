from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision.transforms import Compose, Normalize, ToTensor

TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


class TinyImageNet(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root, self.split = Path(root), split
        self.transform = transform or Compose([
            ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        wnids = [x.strip() for x in (self.root / "wnids.txt").read_text().splitlines()]
        self.class_to_idx = {w: i for i, w in enumerate(wnids)}
        if split == "train":
            self.samples = [(p, self.class_to_idx[w]) for w in wnids
                            for p in sorted((self.root / "train" / w / "images").glob("*.JPEG"))]
        elif split == "val":
            ann = {}
            for line in (self.root / "val" / "val_annotations.txt").read_text().splitlines():
                fields = line.split("\t")
                ann[fields[0]] = self.class_to_idx[fields[1]]
            self.samples = [(self.root / "val" / "images" / name, label)
                            for name, label in sorted(ann.items())]
        else:
            raise ValueError("split must be train or val")
        if not self.samples:
            raise RuntimeError(f"no {split} images found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), target


def ensure_dataset(root, download=False):
    root = Path(root)
    if (root / "wnids.txt").exists():
        return root
    if not download:
        raise FileNotFoundError(f"Tiny-ImageNet not found at {root}")
    archive = root.parent / "tiny-imagenet-200.zip"
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(TINY_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root.parent)
    except Exception as exc:
        raise RuntimeError(f"failed to download Tiny-ImageNet: {exc}") from exc
    return root


class EpochBatchSampler(Sampler[list[int]]):
    """Deterministic, resumable batches: epoch e uses data_seed+e."""
    def __init__(self, size, batch_size, data_seed, drop_last=True, epoch=0, batch_offset=0):
        self.size, self.batch_size, self.data_seed = size, batch_size, data_seed
        self.drop_last, self.epoch, self.batch_offset = drop_last, epoch, batch_offset

    @property
    def batches_per_epoch(self):
        return self.size // self.batch_size if self.drop_last else (
            self.size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.data_seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        for b in range(self.batch_offset, self.batches_per_epoch):
            lo, hi = b * self.batch_size, min((b + 1) * self.batch_size, self.size)
            if hi - lo == self.batch_size or not self.drop_last:
                yield order[lo:hi]

    def __len__(self):
        return self.batches_per_epoch - self.batch_offset

    def advance(self):
        self.batch_offset += 1
        if self.batch_offset >= self.batches_per_epoch:
            self.epoch += 1
            self.batch_offset = 0

    def state_dict(self):
        return {"epoch": self.epoch, "batch_offset": self.batch_offset}

    def load_state_dict(self, state):
        self.epoch, self.batch_offset = state["epoch"], state["batch_offset"]


def make_train_loader(dataset, micro_batch_size, data_seed, num_workers=0, pin_memory=False,
                      drop_last=True, sampler_state=None):
    sampler = EpochBatchSampler(len(dataset), micro_batch_size, data_seed, drop_last)
    if sampler_state:
        sampler.load_state_dict(sampler_state)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                      pin_memory=pin_memory), sampler


def select_probe_indices(size, probe_size, seed, output_path=None):
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed))[
        :min(size, probe_size)].tolist()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(indices, indent=2))
    return indices


def infinite_batches(loader_factory, sampler):
    """Yield forever while advancing sampler immediately after each yielded batch."""
    while True:
        yielded = False
        for batch in loader_factory(sampler):
            yielded = True
            sampler.advance()
            yield batch
        if not yielded:
            raise RuntimeError("sampler produced no batches")
