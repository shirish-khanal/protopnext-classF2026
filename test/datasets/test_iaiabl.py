"""Tests for the IAIA-BL dataset loaders ("iaiabl_fa" and "iaiabl_finer"), like the tests of
protopnext's other loaders in this folder.

A tiny fake IAIA-BL folder is built from PNGs: 2 classes, masks where 255 = outside
the lesion and 0 = the lesion box. Masked batches carry the mask as "fine_annotation"
(1 outside the lesion), the single key protopnext's fine-annotation loss and the
activation precision metric both read.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from protopnet.datasets import iaiabl_fa, iaiabl_finer, training_dataloaders

SIZE = (8, 8)
CLASSES = ["a", "b"]
BATCHES = {"train": 2, "project": 2, "val": 2}


def write_split(root, split, n_per_class, masks=True):
    rng = np.random.default_rng(0)
    for c in CLASSES:
        (root / split / c).mkdir(parents=True, exist_ok=True)
        if masks:
            (root / f"{split}_masks" / c).mkdir(parents=True, exist_ok=True)
        for k in range(n_per_class):
            name = f"{c}{k}.png"
            Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(
                root / split / c / name
            )
            if masks:
                mask = np.full((16, 16), 255, dtype=np.uint8)
                mask[:8, :8] = 0  # lesion box: top-left quarter
                Image.fromarray(mask).save(root / f"{split}_masks" / c / name)


@pytest.fixture
def iaiabl_dir(tmp_path):
    write_split(tmp_path, "train", 3)
    write_split(tmp_path, "train_finer", 1)
    write_split(tmp_path, "train_original", 2, masks=False)
    write_split(tmp_path, "validation", 2)
    write_split(tmp_path, "test_DONOTTOUCH", 1, masks=False)
    return tmp_path


def loaders(name, data_path, **kw):
    return training_dataloaders(
        name,
        data_path=data_path,
        image_size=SIZE,
        batch_sizes=dict(BATCHES),
        num_workers=0,
        **kw,
    )


def test_masked_image_folder_yields_the_mask_as_fine_annotation(iaiabl_dir):
    ds = iaiabl_fa.MaskedImageFolder(
        iaiabl_dir / "validation",
        iaiabl_dir / "validation_masks",
        SIZE,
        (0.5,) * 3,
        (0.5,) * 3,
    )
    item = ds[0]
    assert set(item) == {"img", "target", "sample_id", "fine_annotation"}
    mask = item["fine_annotation"]
    assert mask.shape == (1, *SIZE)
    assert set(mask.unique().tolist()) == {0.0, 1.0}
    assert mask[0, 0, 0] == 0.0 and mask[0, -1, -1] == 1.0  # 0 on the lesion, 1 outside
    assert ds.classes == CLASSES


@pytest.mark.parametrize("name", ["iaiabl_fa", "iaiabl_finer"])
def test_bundle_has_what_protopnext_training_and_proto_rset_need(iaiabl_dir, name):
    bundle = loaders(name, iaiabl_dir)
    assert bundle.num_classes == 2 and bundle.image_size == SIZE
    assert dict(bundle.class_name_ref_dict) == {0: "a", 1: "b"}
    assert len(bundle.train_distribution.mean) == 3
    assert bundle.project_loader is bundle.train_loader_no_aug
    assert len(bundle.project_loader.dataset) == 4  # train_original
    assert "fine_annotation" not in bundle.project_loader.dataset[0]

    batch = next(iter(bundle.train_loader))
    assert batch["fine_annotation"].shape == (2, 1, *SIZE)
    assert "fine_anno" not in batch
    val = next(iter(bundle.val_loader))
    assert val["fine_annotation"].shape == (2, 1, *SIZE)
    assert bundle.val_loader.dataset.classes == CLASSES

    # masked images use the normalization protopnext computed from the training images
    normalize = bundle.val_loader.dataset.image_transform.transforms[-1]
    assert torch.allclose(
        torch.as_tensor(normalize.mean), torch.as_tensor(bundle.train_distribution.mean)
    )
    assert torch.allclose(
        torch.as_tensor(normalize.std), torch.as_tensor(bundle.train_distribution.std)
    )


@pytest.mark.parametrize("name", ["iaiabl_fa", "iaiabl_finer"])
def test_a_split_without_masks_uses_the_standard_loader(iaiabl_dir, name):
    bundle = loaders(name, iaiabl_dir, val_dir="test_DONOTTOUCH")
    assert "fine_annotation" not in bundle.val_loader.dataset[0]
    assert bundle.val_loader.dataset.classes == CLASSES


def test_data_path_defaults_to_the_iaiabl_dir_environment_variable(
    iaiabl_dir, monkeypatch
):
    # both datasets resolve the folder with iaiabl_fa.resolve_data_path
    monkeypatch.setenv("IAIABL_DIR", str(iaiabl_dir))
    bundle = iaiabl_finer.train_dataloaders(
        image_size=SIZE, batch_sizes=dict(BATCHES), num_workers=0
    )
    assert bundle.num_classes == 2


def test_finer_training_samples_finer_images_at_the_requested_fraction(iaiabl_dir):
    bundle = loaders("iaiabl_finer", iaiabl_dir)
    sampler = bundle.train_loader.sampler
    n_main, n_finer, f = 6, 2, iaiabl_finer.FINER_FRACTION
    weights = sampler.weights.tolist()
    assert weights[:n_main] == pytest.approx([(1 - f) / n_main] * n_main)
    assert weights[n_main:] == pytest.approx([f / n_finer] * n_finer)
    assert sampler.num_samples == int(n_main / (1 - f))
