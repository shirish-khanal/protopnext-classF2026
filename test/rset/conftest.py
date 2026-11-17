from dataclasses import dataclass, field
from typing import Tuple

import matplotlib

# The tests draw figures, some in background threads (the factory's analyses); the macOS display
# backend aborts Python there, so use the non-display backend, as on a headless cluster.
matplotlib.use("Agg")

import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from protopnet.activations import CosPrototypeActivation  # noqa: E402
from protopnet.backbones import construct_backbone  # noqa: E402
from protopnet.datasets.dataset_prep import SplitDistribution  # noqa: E402
from protopnet.embedding import AddonLayers  # noqa: E402
from protopnet.models.vanilla_protopnet import VanillaProtoPNet  # noqa: E402
from protopnet.prediction_heads import LinearClassPrototypePredictionHead  # noqa: E402


class TensorDatasetDict(TensorDataset):
    """TensorDataset that yields protopnext-style dict samples."""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return {"img": img, "target": target, "sample_id": index}


def make_tiny_ppnet(activation=None) -> VanillaProtoPNet:
    """Unprojected 2-class VanillaProtoPNet with 2 prototypes per class.

    Prototypes 0 and 1 belong to class 0; prototypes 2 and 3 to class 1.
    """
    if activation is None:
        activation = CosPrototypeActivation()
    backbone = construct_backbone("resnet18")
    ppnet = VanillaProtoPNet(
        backbone=backbone,
        add_on_layers=AddonLayers(
            num_prototypes=4,
            input_channels=backbone.latent_dimension[0],
            proto_channels=128,
        ),
        activation=activation,
        num_classes=2,
        num_prototypes_per_class=2,
    )
    ppnet.prototype_prediction_head = LinearClassPrototypePredictionHead(
        prototype_class_identity=ppnet.prototype_layer.prototype_class_identity.clone(),
        incorrect_class_connection=0,
    )
    return ppnet.eval()


@dataclass
class FakeSplitLoaders:
    project_loader: DataLoader
    val_loader: DataLoader
    train_distribution: SplitDistribution = field(
        default_factory=lambda: SplitDistribution(
            mean=torch.zeros(3), std=torch.ones(3)
        )
    )
    image_size: Tuple[int, int] = (224, 224)


@pytest.fixture
def toy_loader():
    data = torch.randn(6, 3, 224, 224)
    labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.long)
    return DataLoader(TensorDatasetDict(data, labels), batch_size=6, shuffle=False)


@pytest.fixture
def toy_batch(toy_loader):
    return next(iter(toy_loader))


@pytest.fixture
def tiny_ppnet():
    return make_tiny_ppnet()


@pytest.fixture
def projected_ppnet(tiny_ppnet, toy_loader):
    tiny_ppnet.project(toy_loader)
    return tiny_ppnet


@pytest.fixture
def split_loaders(toy_loader):
    return FakeSplitLoaders(project_loader=toy_loader, val_loader=toy_loader)


@pytest.fixture
def saved_ppnet_path(tiny_ppnet, temp_dir):
    path = temp_dir / "ppnet.pth"
    torch.save(tiny_ppnet, path)
    return path


@pytest.fixture
def saved_projected_ppnet_path(projected_ppnet, temp_dir):
    path = temp_dir / "projected_ppnet.pth"
    torch.save(projected_ppnet, path)
    return path


# ---- shared by the metric and figure tests: 8x8 activation maps and a last layer

H = W = 8


class LastLayerHead(torch.nn.Module):
    """A prediction head holding only the last layer, with the given (classes, prototypes) weight."""

    def __init__(self, weight):
        super().__init__()
        self.class_connection_layer = torch.nn.Linear(
            weight.shape[1], weight.shape[0], bias=False
        )
        self.class_connection_layer.weight.data = weight


def act_map(rows, cols):
    a = torch.zeros(H, W)
    a[rows, cols] = 1.0
    return a


# 4 of 64 pixels active (just over the top 5%), inside / outside a 4x4 lesion in the top-left corner
INSIDE = act_map(slice(0, 2), slice(0, 2))
OUTSIDE = act_map(slice(6, 8), slice(6, 8))


# ---- figure stubs: prototype j's activation map is channel j of the image


class ChannelsAsActivations(torch.nn.Module):
    """Prototype j's activation map is channel j of the input image."""

    def __init__(self, weight):
        super().__init__()
        self.prototype_prediction_head = LastLayerHead(weight)

    def forward(self, img, return_prototype_layer_output_dict=False):
        return {"prototype_activations": img}


class Dataset(list):
    """A list of items, optionally with .classes like an ImageFolder."""

    def __init__(self, items, classes=None):
        super().__init__(items)
        if classes is not None:
            self.classes = classes


def lesion_mask():
    lesion = torch.zeros(H, W, dtype=torch.bool)
    lesion[:4, :4] = True
    return lesion


def item(channels, target, mask=None):
    if mask is None:
        mask = (~lesion_mask()).float().unsqueeze(0)  # 0 on the lesion
    return {
        "img": torch.stack(channels),
        "target": torch.tensor(target),
        "fine_annotation": mask,
    }


# Prototypes 0 and 1 belong to class 0, prototype 2 to class 1.
W_BEFORE = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


# The edit removed prototype 1 (its last-layer column is now all zeros).
W_AFTER = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


BEFORE, AFTER = ChannelsAsActivations(W_BEFORE), ChannelsAsActivations(W_AFTER)


ITEMS = [
    item([INSIDE, OUTSIDE, OUTSIDE], 0),  # image 0: removing proto 1 helps
    item([OUTSIDE, INSIDE, OUTSIDE], 0),  # image 1: removing proto 1 hurts
    item([INSIDE, INSIDE, INSIDE], 1),  # image 2: class 1 was not edited
]


DS = Dataset(ITEMS, classes=["cat", "dog"])
