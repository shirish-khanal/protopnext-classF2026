"""
Loader for the IAIA-BL dataset with fine-annotation masks (dataset name "iaiabl_fa").
Ported from proto-rset-iaiabl.

Expected layout (e.g. Lo1136i_paper_fa_png):
    train/<class>/<id>.png              augmented images (training)
    train_masks/<class>/<id>.png        their masks (255 = outside the lesion, 0 = lesion box)
    train_original/<class>/<id>.png     original images (projection, Proto-RSet)
    validation/ and validation_masks/   validation images and masks
    test_DONOTTOUCH/                    test images (no masks needed)

Training and validation batches contain "fine_annotation" (1 x H x W, 1 = outside the lesion),
the key protopnext's fine-annotation loss and activation precision read. Images use the same
resize as the standard loader and the normalization protopnext computes from the training
images, with no extra augmentation (already done offline). The data folder is `data_path`,
or the IAIABL_DIR environment variable when it is not given (as protopnext training does).
"""

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from protopnet.visualization import KeyReturningDict

from .torch_extensions import FilesystemSplitDataloaders


class LoaderBundle:
    """
    The loaders and metadata an IAIA-BL dataset module returns. protopnext has no shared class
    for this; these are proto-rset's LoaderBundle attributes plus `train_distribution`, which
    protopnext's dataloaders and ProtoRSetFactory use for the normalization statistics.
    """

    def __init__(
        self,
        train_dl,
        train_loader_no_aug,
        val_dl,
        proj_dl,
        image_size,
        num_classes,
        train_distribution,
        debug_forbid_loader=None,
        debug_remember_loader=None,
        class_name_ref_dict=KeyReturningDict(),
    ):
        self.train_loader = train_dl
        self.train_loader_no_aug = train_loader_no_aug
        self.project_dataloader = proj_dl
        self.project_loader = proj_dl
        self.val_dataloader = val_dl
        self.val_loader = val_dl
        self.num_classes = num_classes
        self.debug_forbid_loader = debug_forbid_loader
        self.debug_remember_loader = debug_remember_loader
        self.image_size = image_size
        self.train_distribution = train_distribution
        self.normalize_mean = train_distribution.mean
        self.normalize_std = train_distribution.std
        self.class_name_ref_dict = class_name_ref_dict


def resolve_data_path(data_path):
    """`data_path`, else the IAIABL_DIR environment variable."""
    if data_path is None:
        data_path = os.environ.get("IAIABL_DIR")
    if data_path is None:
        raise ValueError(
            "No IAIA-BL data folder: pass data_path or set the IAIABL_DIR environment variable."
        )
    return Path(data_path)


class MaskedImageFolder(ImageFolder):
    """Images from root/<class>/<name>.png, each with the mask mask_root/<class>/<name>.png."""

    def __init__(self, root, mask_root, image_size, mean, std):
        super().__init__(str(root))
        self.mask_root = Path(mask_root)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(size=image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        # Nearest-neighbour resizing keeps the mask strictly 0 / 1
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=image_size, interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.ToTensor(),
            ]
        )
        missing = sum(not self._mask_path(p).exists() for p, _ in self.samples)
        if missing:
            raise FileNotFoundError(
                f"{missing} of {len(self.samples)} images in {root} have no mask in {mask_root}"
            )

    def _mask_path(self, image_path):
        p = Path(image_path)
        return self.mask_root / p.parent.name / p.name

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.image_transform(default_loader(path))
        mask = self.mask_transform(Image.open(self._mask_path(path)).convert("L"))
        return {
            "img": img,
            "target": target,
            "sample_id": index,
            "fine_annotation": (mask > 0.5).float(),
        }


def masked_dataloaders(
    data_path,
    train_dir,
    val_dir,
    image_size,
    batch_sizes,
    num_workers,
    make_train_loader,
):
    """
    The LoaderBundle shared by the IAIA-BL dataset modules. Only training differs between them:
    make_train_loader(masked, loader, class_names) returns the training DataLoader, where
    masked(split) is the split's MaskedImageFolder and loader(dataset, batch_size, **kwargs)
    builds a DataLoader with this module's worker settings.
    """
    data_path = resolve_data_path(data_path)
    class_names = sorted(
        d.name for d in (data_path / train_dir).iterdir() if d.is_dir()
    )

    # Standard loaders on the ORIGINAL images: projection, the unaugmented loader (Proto-RSet),
    # and validation/test when the split has no masks
    base = FilesystemSplitDataloaders(
        data_path=str(data_path),
        num_classes=len(class_names),
        image_size=image_size,
        batch_sizes=batch_sizes,
        train_dir="train_original",
        val_dir=val_dir,
    )
    mean, std = base.train_distribution.mean, base.train_distribution.std

    def masked(split):
        return MaskedImageFolder(
            data_path / split, data_path / f"{split}_masks", image_size, mean, std
        )

    def loader(dataset, batch_size, **kwargs):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs,
        )

    # Validation/test: with masks when a <split>_masks folder exists
    if (data_path / f"{val_dir}_masks").is_dir():
        val_dl = loader(masked(val_dir), batch_sizes["val"], shuffle=False)
    else:
        val_dl = base.val_loader

    return LoaderBundle(
        train_dl=make_train_loader(masked, loader, class_names),
        train_loader_no_aug=base.project_loader,  # protopnext's project loader: train_original, no augmentation
        val_dl=val_dl,
        proj_dl=base.project_loader,
        num_classes=len(class_names),
        image_size=image_size,
        train_distribution=base.train_distribution,
        class_name_ref_dict=KeyReturningDict(dict(enumerate(class_names))),
    )


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
        return loader(masked(train_dir), batch_sizes["train"], shuffle=True)

    return masked_dataloaders(
        data_path,
        train_dir,
        val_dir,
        image_size,
        batch_sizes,
        num_workers,
        train_loader,
    )
