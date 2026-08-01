from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_csv(row: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_csv_rows(rows, path):
    """Append a homogeneous batch of rows with a single file open."""
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows[1:]):
        raise ValueError("CSV batch rows must have identical fields in identical order")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with open(path, newline="", encoding="utf-8") as f:
            existing_fields = next(csv.reader(f), None)
        if existing_fields != fieldnames:
            raise ValueError(
                f"CSV schema mismatch for {path}: "
                f"existing={existing_fields}, new={fieldnames}")
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def truncate_csv_after_step(path, max_step: int, column: str = "global_state_step",
                            dedupe_by=()):
    """Atomically discard post-checkpoint rows and optional duplicate records."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = reader.fieldnames
            required = {column, *dedupe_by}
            if fields is None or not required.issubset(fields):
                raise ValueError(f"{path} is missing CSV fields {sorted(required)}")
            seen = set()
            with open(tmp, "w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=fields)
                writer.writeheader()
                for row in reader:
                    if int(row[column]) > int(max_step):
                        continue
                    key = tuple(row[name] for name in dedupe_by)
                    if dedupe_by and key in seen:
                        continue
                    seen.add(key)
                    writer.writerow(row)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def last_csv_values(path, key_column: str, value_column: str, **equals):
    """Return each key's last numeric value from an optionally filtered CSV."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    result = {}
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        required = {key_column, value_column, *equals}
        if not required.issubset(fields):
            raise ValueError(
                f"{path} is missing CSV fields {sorted(required - fields)}")
        expected = {key: str(value) for key, value in equals.items()}
        for row in reader:
            if any(row[key] != value for key, value in expected.items()):
                continue
            try:
                result[row[key_column]] = float(row[value_column])
            except ValueError as exc:
                raise ValueError(
                    f"non-numeric {value_column} in {path}: "
                    f"{row[value_column]!r}") from exc
    return result


def set_all_seeds(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def capture_rng_states():
    return {
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_states(state):
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if torch.cuda.is_available() and state.get("torch_cuda_rng_all"):
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_all"])


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def environment_info():
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "os": platform.platform(), "git_commit": git_commit(),
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_gradients(model):
    return all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())

