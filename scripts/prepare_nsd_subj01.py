#!/usr/bin/env python3
"""Build the Subject 1 metadata and beta HDF5 expected by NeuroAdapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from neuroadapter_research.atomic import write_json_atomic


TRIALS_PER_SESSION = 750
SESSIONS = 40
TOTAL_TRIALS = TRIALS_PER_SESSION * SESSIONS
VERTICES_PER_HEMISPHERE = 163842


def summarize_nonfinite(array: np.ndarray, path: Path) -> dict[str, object] | None:
    finite = np.isfinite(array)
    if finite.all():
        return None
    invalid = ~finite
    trial_indices, vertex_indices = np.nonzero(invalid)
    return {
        "source_path": str(path),
        "nonfinite_count": int(invalid.sum()),
        "nan_count": int(np.isnan(array).sum()),
        "positive_inf_count": int(np.isposinf(array).sum()),
        "negative_inf_count": int(np.isneginf(array).sum()),
        "affected_trial_count": int(np.unique(trial_indices).size),
        "affected_vertex_count": int(np.unique(vertex_indices).size),
        "affected_trial_indices_zero_based": np.unique(trial_indices).tolist(),
        "affected_vertex_indices_zero_based": np.unique(vertex_indices).tolist(),
    }


def load_mgh_session(path: Path) -> tuple[np.ndarray, dict[str, object] | None]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    array = np.asarray(data[:, 0, 0, :], dtype=np.float32).T
    expected = (TRIALS_PER_SESSION, VERTICES_PER_HEMISPHERE)
    if array.shape != expected:
        raise ValueError(f"{path}: expected {expected}, got {array.shape}")
    return array, summarize_nonfinite(array, path)


def build_metadata(nsd_root: Path, output_dir: Path, subject: int) -> dict[str, np.ndarray]:
    design = sio.loadmat(nsd_root / "experiments" / "nsd" / "nsd_expdesign.mat")
    subject_images = np.asarray(design["subjectim"], dtype=np.int64)[subject - 1]
    master_ordering = np.asarray(design["masterordering"], dtype=np.int64).reshape(-1)
    shared_indices = np.asarray(design["sharedix"], dtype=np.int64).reshape(-1)

    presentation_order = subject_images[master_ordering - 1] - 1
    all_subject_images = subject_images - 1
    test_images = np.unique(shared_indices - 1)
    train_images = np.setdiff1d(all_subject_images, test_images, assume_unique=False)

    if presentation_order.shape != (TOTAL_TRIALS,):
        raise ValueError(f"expected {TOTAL_TRIALS} presentations, got {presentation_order.shape}")
    unique_images, repeat_counts = np.unique(presentation_order, return_counts=True)
    if unique_images.size != 10000 or not np.all(repeat_counts == 3):
        raise ValueError("expected 10000 unique images with exactly three presentations each")
    if train_images.size != 9000 or test_images.size != 1000:
        raise ValueError(f"expected train/test 9000/1000, got {train_images.size}/{test_images.size}")
    if np.intersect1d(train_images, test_images).size:
        raise ValueError("train and test image IDs overlap")

    beta_dir = (
        nsd_root
        / "betas"
        / f"subj{subject:02}"
        / "fsaverage"
        / "betas_fithrf_GLMdenoise_RR"
    )
    metadata: dict[str, np.ndarray] = {
        "img_presentation_order": presentation_order.astype(np.int64),
        "train_img_num": train_images.astype(np.int64),
        "test_img_num": test_images.astype(np.int64),
        "val_img_num": np.empty(0, dtype=np.int64),
    }

    for hemi in ("lh", "rh"):
        path = beta_dir / f"{hemi}.ncsnr.mgh"
        ncsnr = nib.load(str(path)).get_fdata(dtype=np.float32)
        ncsnr = np.asarray(ncsnr[:, 0, 0], dtype=np.float32)
        if ncsnr.shape != (VERTICES_PER_HEMISPHERE,) or not np.isfinite(ncsnr).all():
            raise ValueError(f"invalid {hemi} ncsnr: shape={ncsnr.shape}")
        metadata[f"{hemi}_ncsnr"] = ncsnr

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"metadata_sub-{subject:02}.npy"
    temporary = output_dir / f"metadata_sub-{subject:02}.npy.tmp"
    with temporary.open("wb") as handle:
        np.save(handle, metadata, allow_pickle=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return metadata


def build_betas(nsd_root: Path, output_dir: Path, subject: int, overwrite: bool) -> None:
    target = output_dir / f"betas_sub-{subject:02}.h5"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists; pass --overwrite to rebuild")

    beta_dir = (
        nsd_root
        / "betas"
        / f"subj{subject:02}"
        / "fsaverage"
        / "betas_fithrf_GLMdenoise_RR"
    )
    temporary = target.with_suffix(".h5.tmp")
    temporary.unlink(missing_ok=True)

    anomalies: list[dict[str, object]] = []
    with h5py.File(temporary, "w") as h5:
        h5.attrs["subject"] = subject
        h5.attrs["surface_space"] = "fsaverage"
        h5.attrs["beta_derivative"] = "betas_fithrf_GLMdenoise_RR"
        h5.attrs["source_nonfinite_values_preserved"] = True
        for hemi in ("lh", "rh"):
            dataset = h5.create_dataset(
                f"{hemi}_betas",
                shape=(TOTAL_TRIALS, VERTICES_PER_HEMISPHERE),
                dtype="float32",
                chunks=(8, VERTICES_PER_HEMISPHERE),
                compression="lzf",
            )
            cursor = 0
            for session in tqdm(range(1, SESSIONS + 1), desc=f"{hemi} sessions"):
                source = beta_dir / f"{hemi}.betas_session{session:02}.mgh"
                array, anomaly = load_mgh_session(source)
                if anomaly is not None:
                    anomaly["hemisphere"] = hemi
                    anomaly["session"] = session
                    anomalies.append(anomaly)
                dataset[cursor : cursor + TRIALS_PER_SESSION] = array
                cursor += TRIALS_PER_SESSION
            if cursor != TOTAL_TRIALS:
                raise RuntimeError(f"{hemi}: wrote {cursor} trials instead of {TOTAL_TRIALS}")
        h5.flush()

    os.replace(temporary, target)
    report = {
        "schema_version": 1,
        "subject": subject,
        "policy": (
            "Preserve source values exactly. Formal training is permitted only after a "
            "full scan proves that all vertices in the selected top-SNR parcels are finite."
        ),
        "anomalous_file_count": len(anomalies),
        "nonfinite_value_count": sum(int(item["nonfinite_count"]) for item in anomalies),
        "files": anomalies,
    }
    write_json_atomic(output_dir / "source_nonfinite_values.json", report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsd-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = build_metadata(args.nsd_root, args.output_dir, args.subject)
    print(f"presentations={metadata['img_presentation_order'].size}")
    print(f"train={metadata['train_img_num'].size}")
    print(f"test={metadata['test_img_num'].size}")
    print(f"validation={metadata['val_img_num'].size}")
    if not args.metadata_only:
        build_betas(args.nsd_root, args.output_dir, args.subject, args.overwrite)


if __name__ == "__main__":
    main()
