"""Tests for the activation precision metric in protopnet/metrics.py and its logging during training.

A stub model returns hand-made activation maps, so each expected precision can be
worked out by hand: precision = share of the top 5% of a prototype's activation map
that lies on the lesion, where lesion = 1 - fine_annotation. The metric works for any
dataset: class names come from the data. Like protopnext's fine-annotation loss it does
not validate the data; callers use it only when fine annotation is on (fa_coef set). Training
logs it through InterpretableTrainingMetrics exactly when fine annotation is on.

"""

import math
from types import SimpleNamespace

import pytest
import torch

import protopnet.train_protopnet as train_protopnet
from protopnet.metrics import (
    ActivationPrecisionScore,
    activation_precision_score,
)
from protopnet.train.metrics import InterpretableTrainingMetrics
from protopnet.train.trainer import format_metrics
from protopnet.utilities.general_utilities import class_names_from

from .conftest import INSIDE, OUTSIDE, H, LastLayerHead, W


class StubModel(torch.nn.Module):
    """Returns fixed activation maps (N, P, H, W) for a single batch of N images."""

    def __init__(self, weight, acts):
        super().__init__()
        self.prototype_prediction_head = LastLayerHead(weight)
        self.acts = acts

    def forward(self, img, return_prototype_layer_output_dict=False):
        return {"prototype_activations": self.acts}


class MustNotRun(StubModel):
    def forward(self, *args, **kwargs):
        raise AssertionError("the model must not run when no image is annotated")


class FakeDataset:
    def __init__(self, n, classes=None):
        self.n = n
        if classes is not None:
            self.classes = classes

    def __len__(self):
        return self.n


class Loader(list):
    """A list of batches with a .dataset, like a DataLoader."""

    def __init__(self, batches, dataset):
        super().__init__(batches)
        self.dataset = dataset


def lesion_top_left():
    fine_annotation = torch.ones(1, H, W)
    fine_annotation[0, :4, :4] = 0  # fine_annotation is 0 on the lesion
    return fine_annotation


CLASSES = ["cat", "dog", "bird"]


def make_case(targets, acts, weight, classes=CLASSES, masks=None, model_cls=StubModel):
    """One batch of len(targets) images; acts is (N, P, H, W); weight is (classes, P)."""
    if masks is None:
        masks = torch.stack([lesion_top_left() for _ in targets])
    batch = {
        "img": torch.zeros(len(targets), 3, H, W),
        "target": torch.tensor(targets),
        "fine_annotation": masks,
    }
    return model_cls(weight, acts), Loader([batch], FakeDataset(len(targets), classes))


def two_images_three_prototypes(**kw):
    # prototype 0 -> class 0, prototype 1 -> class 1 (coef 2.0),
    # prototype 2 -> no class (its largest last-layer weight is not > 0)
    weight = torch.tensor([[1.0, 0.0, -1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    acts = torch.stack(
        [
            torch.stack([INSIDE, OUTSIDE, INSIDE]),  # image 0, class 0
            torch.stack([OUTSIDE, OUTSIDE, INSIDE]),  # image 1, class 1
        ]
    )
    return make_case([0, 1], acts, weight, **kw)


def weight_of(model):
    return model.prototype_prediction_head.class_connection_layer.weight


# ---- activation_precision_score(): one score by default, or by class / by prototype


def test_only_prototypes_of_the_image_class_are_scored():
    model, loader = two_images_three_prototypes()
    assert activation_precision_score(model, loader, "cpu") == 0.5
    assert activation_precision_score(model, loader, "cpu", by="prototype") == {
        0: ("cat", 1.0, 1.0, 1),
        1: ("dog", 2.0, 0.0, 1),
    }


def test_score_by_class_covers_every_class_of_the_data():
    model, loader = two_images_three_prototypes()
    by_class = activation_precision_score(model, loader, "cpu", by="class")
    assert list(by_class) == CLASSES
    assert by_class["cat"] == (1.0, 1)
    assert by_class["dog"] == (0.0, 1)
    mean, n = by_class["bird"]
    assert math.isnan(mean) and n == 0


def test_class_name_count_must_match_the_model():
    with pytest.raises(ValueError, match="2 class names but the model has 3 classes"):
        class_names_from(FakeDataset(1, ["cat", "dog"]), num_classes=3)


def test_images_with_an_empty_mask_are_skipped():
    masks = torch.stack(
        [lesion_top_left(), torch.zeros(1, H, W)]
    )  # image 1: no annotation
    model, loader = two_images_three_prototypes(masks=masks)
    assert activation_precision_score(model, loader, "cpu", by="prototype") == {
        0: ("cat", 1.0, 1.0, 1)
    }


def test_all_masks_empty_gives_nan_without_running_the_model():
    model, loader = two_images_three_prototypes(
        masks=torch.zeros(2, 1, H, W), model_cls=MustNotRun
    )
    assert math.isnan(activation_precision_score(model, loader, "cpu"))


def test_sparse_masks_give_the_same_result():
    dense_model, dense_loader = two_images_three_prototypes()
    sparse_model, sparse_loader = two_images_three_prototypes()
    sparse_loader[0]["fine_annotation"] = sparse_loader[0][
        "fine_annotation"
    ].to_sparse()
    for by in (None, "prototype"):
        assert activation_precision_score(
            sparse_model, sparse_loader, "cpu", by=by
        ) == (activation_precision_score(dense_model, dense_loader, "cpu", by=by))


# ---- ActivationPrecisionScore: the torchmetrics Metric behind it (used in training)


def test_metric_accumulates_batches_and_breaks_down_by_index():
    model, loader = two_images_three_prototypes()
    batch, acts, weight = loader[0], model.acts, weight_of(model)
    metric = ActivationPrecisionScore()
    for i in (0, 1):  # one image per batch
        metric.update(
            acts[i : i + 1],
            batch["target"][i : i + 1],
            batch["fine_annotation"][i : i + 1],
            weight,
        )

    assert metric.compute().item() == 0.5
    assert metric.per_prototype() == {0: (0, 1.0, 1.0, 1), 1: (1, 2.0, 0.0, 1)}
    per_class = metric.per_class()
    assert per_class[0] == (1.0, 1) and per_class[1] == (0.0, 1)
    assert math.isnan(per_class[2][0]) and per_class[2][1] == 0


# ---- logged during training (InterpretableTrainingMetrics), gated on fine annotation


class ProtoPNetStub:
    """Prototype 0 belongs to class 0, prototype 1 to class 1."""

    def __init__(self):
        layer = torch.nn.Linear(2, 2, bias=False)
        layer.weight.data = torch.eye(2)
        self.prototype_prediction_head = SimpleNamespace(class_connection_layer=layer)

    def prototypes_embedded(self):
        return False


def training_batch(images):
    """images: list of (target, [map of proto 0, map of proto 1])."""
    mask = torch.ones(1, H, W)
    mask[0, :4, :4] = 0  # 0 on the lesion
    targets = torch.tensor([t for t, _ in images])
    forward_args = {
        "img": torch.zeros(len(images), 3, H, W),
        "target": targets,
        "fine_annotation": torch.stack([mask] * len(images)),
    }
    forward_outputs = {
        "logits": torch.nn.functional.one_hot(targets, 2).float(),
        "prototype_activations": torch.stack([torch.stack(maps) for _, maps in images]),
    }
    return forward_args, forward_outputs


def training_metrics(**kw):
    return InterpretableTrainingMetrics(
        protopnet=ProtoPNetStub(),
        num_classes=2,
        proto_per_class=1,
        part_num=None,
        img_size=None,
        half_size=None,
        acc_only=True,
        **kw,
    )


TWO_IMAGES = [(0, [INSIDE, OUTSIDE]), (1, [INSIDE, OUTSIDE])]  # precisions 1.0 and 0.0


def test_activation_precision_is_off_by_default():
    metrics = training_metrics()
    metrics.update_all(*training_batch(TWO_IMAGES), phase="eval")
    assert "activation_precision" not in metrics.metrics
    assert "activation_precision" not in metrics.compute_dict()


def test_activation_precision_is_logged_when_fine_annotation_is_on():
    metrics = training_metrics(fine_annotation=True)
    metrics.start_epoch("eval")
    metrics.update_all(*training_batch(TWO_IMAGES), phase="eval")
    # protopnext's per-epoch log line (trainer.format_metrics) prints only plain Python numbers
    assert "activation_precision:0.5" in format_metrics(metrics.compute_dict())


def test_activation_precision_resets_every_epoch():
    metrics = training_metrics(fine_annotation=True)
    metrics.start_epoch("eval")
    metrics.update_all(*training_batch(TWO_IMAGES), phase="eval")
    metrics.compute_dict()

    metrics.start_epoch("eval")
    metrics.update_all(*training_batch([(0, [INSIDE, OUTSIDE])]), phase="eval")
    assert metrics.compute_dict()["activation_precision"] == 1.0


class _Stop(Exception):
    pass


@pytest.mark.parametrize("fa_coef, expected", [(None, False), (0.001, True)])
def test_train_protopnet_turns_activation_precision_on_with_fa_coef(
    monkeypatch, fa_coef, expected
):
    seen = {}

    def capture_metrics(**kwargs):
        seen.update(kwargs)
        raise _Stop  # stop before any training happens

    monkeypatch.setattr(
        train_protopnet, "InterpretableTrainingMetrics", capture_metrics
    )
    monkeypatch.setattr(
        train_protopnet, "WeightsAndBiasesTrainLogger", lambda **kwargs: None
    )
    monkeypatch.setattr(
        train_protopnet.datasets,
        "training_dataloaders",
        lambda *args, **kwargs: SimpleNamespace(
            num_classes=2,
            image_size=(224, 224),
            train_loader=None,
            val_loader=None,
            project_loader=None,
        ),
    )

    with pytest.raises(_Stop):
        train_protopnet.run.callback(
            backbone="resnet18",
            dataset="any",
            fa_coef=fa_coef,
            activation_function="cosine",
        )
    assert seen["fine_annotation"] is expected


def test_the_score_puts_back_each_submodules_own_mode():
    """A submodule kept in eval mode inside a training model (e.g. a frozen backbone) stays in eval mode."""
    model, loader = two_images_three_prototypes()
    model.backbone = torch.nn.BatchNorm2d(1)
    model.train()
    model.backbone.eval()
    activation_precision_score(model, loader)
    assert model.training and not model.backbone.training


class RecordsMode(StubModel):
    def __init__(self, weight, acts):
        super().__init__(weight, acts)
        self.modes = []

    def forward(self, img, return_prototype_layer_output_dict=False):
        self.modes.append(self.training)
        return super().forward(img)


def test_the_score_runs_the_model_in_eval_mode_and_puts_its_mode_back():
    model, loader = two_images_three_prototypes()
    recording = RecordsMode(
        model.prototype_prediction_head.class_connection_layer.weight.data, model.acts
    ).train()
    assert activation_precision_score(recording, loader) == activation_precision_score(
        model, loader, "cpu"
    )
    assert recording.modes == [False] and recording.training
