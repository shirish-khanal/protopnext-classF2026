"""
Loader for IAIA-BL with box masks + finer masks (dataset name "iaiabl_finer").
Ported from proto-rset-iaiabl.

Layout (e.g. Lo1136i_paper_finer_png): train(+_masks) = main augmented (box masks),
train_finer(+_masks) = finer augmented (finer masks), train_original, validation(+_masks),
test_DONOTTOUCH(+_masks), test_finer(+_masks).

Training mixes main and finer images so about FINER_FRACTION of samples are finer
(IAIA-BL appended 10 finer images to each batch of 95, about 9.5%). Masks are in each
batch's "fine_annotation" (see iaiabl_fa), and the data folder is `data_path` or the
IAIABL_DIR environment variable.
"""

import os

from torch.utils.data import ConcatDataset, WeightedRandomSampler

from .iaiabl_fa import masked_dataloaders

FINER_FRACTION = float(os.environ.get("IAIABL_FINER_FRACTION", 10 / 105))


def train_dataloaders(
    data_path=None,
    train_dir="train",
    val_dir="validation",
    image_size=(224, 224),
    batch_sizes={"train": 95, "project": 75, "val": 100},
    num_workers=7,
    **kwargs,
):
    def train_loader(masked, loader, class_names):
        # main (box masks) + finer (finer masks), sampled so ~FINER_FRACTION are finer
        main, finer = masked(train_dir), masked("train_finer")
        assert (
            main.classes == finer.classes == class_names
        ), "class folders differ between main and finer"
        weights = [(1 - FINER_FRACTION) / len(main)] * len(main) + [
            FINER_FRACTION / len(finer)
        ] * len(finer)
        # each main image seen about once per epoch
        num_samples = int(len(main) / (1 - FINER_FRACTION))
        sampler = WeightedRandomSampler(
            weights, num_samples=num_samples, replacement=True
        )
        return loader(
            ConcatDataset([main, finer]), batch_sizes["train"], sampler=sampler
        )

    return masked_dataloaders(
        data_path,
        train_dir,
        val_dir,
        image_size,
        batch_sizes,
        num_workers,
        train_loader,
    )
