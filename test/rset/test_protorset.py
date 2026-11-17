"""Tests for ProtoRSetFactory on protopnext (ported from proto-rset's test/test_rset.py) and for
adding prototypes to a protopnext model (rashomon_sets.prototype_ops)."""
from pathlib import Path

import pytest
import torch

from torch.utils.data import DataLoader

from protopnet.activations import lookup_activation
from protopnet.prototype_layers import PrototypeFromSampleMeta, PrototypeWithMeta
from protopnet.prototypical_part_model import ProtoPNet
from protopnet.utilities.project_utilities import hash_func
from rashomon_sets.protorset_factory import DEFAULT_RSET_ARGS, ProtoRSetFactory
from rashomon_sets.prototype_ops import append_prototypes, prototype_from_latent

from .conftest import TensorDatasetDict, make_tiny_ppnet


def rset_args(**overrides):
    return {
        **DEFAULT_RSET_ARGS,
        "max_iter": 1_000,
        "rashomon_bound_multiplier": 1.1,
        "lam": 0.0,
        "device": "cpu",
        "lr_for_opt": 0.01,
        **overrides,
    }


def make_factory(path, split_loaders, save_dir, **kw):
    return ProtoRSetFactory(
        split_loaders,
        path,
        rashomon_set_args=kw.pop("args", rset_args()),
        device="cpu",
        analysis_save_dir=save_dir,
        **kw,
    )


def model_loss(model, loader, reg_weight=0.0):
    """Cross-entropy of ``model`` on ``loader`` plus an L2 penalty on its last layer."""
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits.append(model(batch["img"])["logits"])
            labels.append(batch["target"])
    weights = model.prototype_prediction_head.class_connection_layer.weight.data
    return torch.nn.functional.cross_entropy(
        torch.cat(logits), torch.cat(labels)
    ).item() + reg_weight * torch.norm(weights).item()


def optimal_rset_loss(factory, reg_weight=0.0):
    features = torch.tensor(factory.train_similarities_dataset.values[:, :-1])
    labels = torch.tensor(factory.train_similarities_dataset.values[:, -1]).long()
    preds = factory.rset.optimal_model(features.to(factory.rset.device)).cpu()
    return torch.nn.functional.cross_entropy(preds, labels).item() + reg_weight * (
        torch.norm(factory.rset.optimal_model.get_params()).item()
    )


def sampled_model_losses(factory, loader, reg_weight=0.0, n=5):
    models = [factory.produce_protopnet_object(return_optimal=False) for _ in range(n)]
    return models, [model_loss(m, loader, reg_weight) for m in models]


def as_groups(paths):
    return paths if isinstance(paths, list) else [paths]


def source_is_in(meta, loader):
    """True if the prototype's recorded source is that loader's image with the same sample id."""
    return meta.hash == hash_func(loader.dataset[int(meta.sample_id)]["img"])


@pytest.fixture
def other_loader():
    """A loader with different images but the same sample ids as toy_loader (like a val split)."""
    data = torch.randn(6, 3, 224, 224)
    labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.long)
    return DataLoader(TensorDatasetDict(data, labels), batch_size=6, shuffle=False)


@pytest.mark.parametrize("activation_name", ["cosine", "l2", "exp_l2"])  # VanillaProtoPNet and TesNet (D7)
def test_rset_loss(activation_name, split_loaders, toy_loader, temp_dir):
    model = make_tiny_ppnet(lookup_activation(activation_name))
    model.project(toy_loader)
    original_loss = model_loss(model, toy_loader)
    path = temp_dir / f"{activation_name}.pth"
    torch.save(model, path)
    factory = make_factory(path, split_loaders, temp_dir / f"loss_{activation_name}")

    _, rset_losses = sampled_model_losses(factory, toy_loader)
    assert optimal_rset_loss(factory) <= original_loss + 1e-3, "'optimal' model loss is too high"
    assert max(rset_losses) <= original_loss * 1.1 + 1e-3, "sampled model loss is too high"


def test_rset_removing_prototypes(projected_ppnet, saved_projected_ppnet_path, split_loaders, toy_loader, temp_dir):
    reg_weight = 1
    original_loss = model_loss(projected_ppnet, toy_loader, reg_weight)
    factory = make_factory(
        saved_projected_ppnet_path, split_loaders, temp_dir / "remove", args=rset_args(lam=reg_weight)
    )

    removal_status = [
        factory.require_to_avoid_prototype(p)
        for p in range(factory.initial_protopnet.prototype_layer.num_prototypes)
    ]
    sampled_models, rset_losses = sampled_model_losses(factory, toy_loader, reg_weight)

    assert optimal_rset_loss(factory, reg_weight) <= original_loss + 1e-3
    assert max(rset_losses) <= original_loss * 1.1 + 1e-3, "sampled model loss is too high"
    assert max(removal_status), "no prototype could be removed"
    for proto_ind, did_remove in enumerate(removal_status):
        if did_remove:
            assert abs(factory.rset.optimal_model.get_params()[proto_ind]) <= 1e-4
            for m in sampled_models:
                weights = m.prototype_prediction_head.class_connection_layer.weight.data
                assert torch.all(abs(weights[:, proto_ind]) <= 1e-4)


def test_prototype_sampling(saved_projected_ppnet_path, split_loaders, toy_loader, temp_dir):
    factory = make_factory(saved_projected_ppnet_path, split_loaders, temp_dir / "sampling")
    model = factory.initial_protopnet
    n = model.prototype_layer.num_prototypes

    factory.sample_additional_prototypes(4, "uniform_random", toy_loader)

    model = factory.initial_protopnet
    assert len(model.prototype_layer.prototype_meta) == n + 4
    assert model.prototype_prediction_head.class_connection_layer.weight.shape[1] == n + 4
    assert model.prototype_prediction_head.prototype_class_identity.shape[0] == n + 4
    assert model.prototype_layer.prototype_tensors.shape[0] == n + 4
    assert all(
        isinstance(m, PrototypeFromSampleMeta) for m in model.prototype_layer.prototype_meta
    )
    for batch in toy_loader:
        assert model(batch["img"]) is not None


def test_factory_projects_unprojected_model(saved_ppnet_path, split_loaders, temp_dir):
    factory = make_factory(saved_ppnet_path, split_loaders, temp_dir / "unprojected")
    assert all(
        isinstance(m, PrototypeFromSampleMeta)
        for m in factory.initial_protopnet.prototype_layer.prototype_meta
    )


def test_factory_skips_project_for_projected_model(
    saved_projected_ppnet_path, split_loaders, temp_dir, monkeypatch
):
    def fail_project(*args, **kwargs):
        raise AssertionError("project() must not run for an already-projected model")

    monkeypatch.setattr(ProtoPNet, "project", fail_project)
    make_factory(saved_projected_ppnet_path, split_loaders, temp_dir / "no_reproject")


def test_factory_leaves_caller_args_untouched(saved_projected_ppnet_path, split_loaders, temp_dir):
    args = rset_args(device=torch.device("cuda"))
    make_factory(saved_projected_ppnet_path, split_loaders, temp_dir / "args", args=args)
    assert args["device"] == torch.device("cuda")
    assert DEFAULT_RSET_ARGS["device"] == torch.device("cuda")


def test_sampling_from_another_loader_keeps_sources_in_projection_loader(
    saved_projected_ppnet_path, split_loaders, toy_loader, other_loader, temp_dir
):
    factory = make_factory(saved_projected_ppnet_path, split_loaders, temp_dir / "other_random")

    factory.sample_additional_prototypes(2, "uniform_random", other_loader)

    meta = factory.initial_protopnet.prototype_layer.prototype_meta
    assert all(source_is_in(m, toy_loader) for m in meta)


def test_sampling_by_index_from_another_loader_keeps_sources_in_projection_loader(
    saved_projected_ppnet_path, split_loaders, toy_loader, other_loader, temp_dir
):
    factory = make_factory(saved_projected_ppnet_path, split_loaders, temp_dir / "other_index")

    factory.sample_prototypes_by_index(other_loader, 4, target_h=0, target_w=0)

    meta = factory.initial_protopnet.prototype_layer.prototype_meta
    assert all(source_is_in(m, toy_loader) for m in meta)


def test_display_methods_write_files(saved_projected_ppnet_path, split_loaders, temp_dir):
    factory = make_factory(saved_projected_ppnet_path, split_loaders, temp_dir / "display")

    for group in as_groups(factory.display_proto_in_class(1)):
        assert all(Path(p).exists() for p in group)
    assert Path(factory.display_local_analysis(0, run_proto_vis=True)[0]).exists()
    views, heatmap = factory.display_global_analysis_for_proto(0)
    assert Path(views).exists() and Path(heatmap).exists()


def test_run_analysis_async_renders_global_and_local_analyses(saved_projected_ppnet_path, split_loaders, toy_loader, temp_dir):
    import asyncio

    factory = make_factory(saved_projected_ppnet_path, split_loaders, temp_dir / "async")
    asyncio.run(factory.run_analysis_async(
        factory.initial_protopnet, saved_projected_ppnet_path, toy_loader, toy_loader, (224, 224), "cpu",
    ))
    n = factory.initial_protopnet.prototype_layer.num_prototypes
    out = factory.vis_save_loc
    assert len(list((out / "prototypes").glob("*_proto_bbox.png"))) == n
    assert len(list((out / "global_analysis_saves").glob("*_view_10.png"))) == n
    assert len(list((out / "local_analysis_saves").glob("*.png"))) == len(toy_loader.dataset)


# ---- adding prototypes to a protopnext model (rashomon_sets.prototype_ops)


def test_append_prototypes_grows_layer_and_head(projected_ppnet, toy_batch):
    model, n = projected_ppnet, projected_ppnet.prototype_layer.num_prototypes
    old_w = model.prototype_prediction_head.class_connection_layer.weight.data.clone()
    latent = model.embed(toy_batch["img"][:1])["latent_vectors"][0]
    img = toy_batch["img"][0]
    append_prototypes(
        model,
        [
            prototype_from_latent(latent, 0, 0, img, sample_id=0, target=1),
            prototype_from_latent(latent, 1, 1, img, sample_id=0, target=0),
        ],
    )
    head = model.prototype_prediction_head
    assert model.prototype_layer.num_prototypes == n + 2
    assert torch.equal(
        model.prototype_layer.prototype_class_identity[n:],
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )
    assert head.class_connection_layer.weight.shape == (2, n + 2)
    assert torch.equal(head.class_connection_layer.weight.data[:, :n], old_w)
    assert torch.equal(
        head.class_connection_layer.weight.data[:, n:],
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )
    assert torch.equal(
        head.prototype_class_identity, model.prototype_layer.prototype_class_identity
    )
    assert head.num_prototypes == n + 2
    assert model.prototype_layer.prototype_meta[n].sample_id == 0
    assert model(toy_batch["img"])["logits"].shape == (6, 2)


def test_append_prototypes_rejects_unprojected_model(tiny_ppnet, toy_batch):
    latent = tiny_ppnet.embed(toy_batch["img"][:1])["latent_vectors"][0]
    with pytest.raises(ValueError, match="project"):
        append_prototypes(
            tiny_ppnet,
            [prototype_from_latent(latent, 0, 0, toy_batch["img"][0], 0, 1)],
        )


def test_prune_after_append_keeps_layers_aligned(projected_ppnet):
    model = projected_ppnet
    model.prune_duplicate_prototypes()  # projection may already create duplicates
    n = model.prototype_layer.num_prototypes
    layer = model.prototype_layer
    append_prototypes(
        model,
        [
            PrototypeWithMeta(
                layer.prototype_tensors[0].detach().clone(), layer.prototype_meta[0]
            )
        ],
    )
    model.prune_duplicate_prototypes()
    assert model.prototype_layer.num_prototypes == n
    assert len(model.prototype_layer.prototype_meta) == n
    assert model.prototype_prediction_head.class_connection_layer.weight.shape == (2, n)
    assert torch.equal(
        model.prototype_prediction_head.prototype_class_identity,
        model.prototype_layer.prototype_class_identity,
    )


# ---- the before/after-edit figures (rashomon_sets.visualization). Stub models use the image's
# channels as their prototype activation maps, so "before" and "after" differ only in last-layer
# weights, like a Proto-RSet edit. Every expected value can be worked out by hand.

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from rashomon_sets.visualization import (  # noqa: E402
    pam_gains,
    plot_before_after,
    plot_prototype_importance,
    prototype_importance_rows,
)

from .conftest import (  # noqa: E402
    AFTER,
    BEFORE,
    DS,
    INSIDE,
    ITEMS,
    OUTSIDE,
    W_AFTER,
    W_BEFORE,
    ChannelsAsActivations,
    Dataset,
    H,
    W,
    item,
    lesion_mask,
)


def _check_pam_gains_rank_images_of_edited_classes():
    results = pam_gains(BEFORE, AFTER, DS, device="cpu")
    assert results == [(0.5, 0, 0.5, 1.0), (-0.5, 1, 0.5, 0.0)]


def _check_pam_gains_skip_images_with_an_empty_mask():
    ds = Dataset(
        [ITEMS[0], item([OUTSIDE, INSIDE, OUTSIDE], 0, mask=torch.zeros(1, H, W))]
    )
    assert pam_gains(BEFORE, AFTER, ds, device="cpu") == [(0.5, 0, 0.5, 1.0)]


def test_pam_gains():
    """Images of edited classes ranked by gain; images with an empty mask skipped."""
    _check_pam_gains_rank_images_of_edited_classes()
    _check_pam_gains_skip_images_with_an_empty_mask()


def test_plot_before_after_draws_the_plain_image_then_three_panels(temp_dir, capsys):
    fig = plot_before_after(
        BEFORE, AFTER, DS, 0, mean=torch.zeros(3), std=torch.ones(3), device="cpu"
    )
    try:
        titles = [ax.get_title() for ax in fig.axes]
        assert len(fig.axes) == 4
        # the first panel is the image alone: no heatmap, outline or box over it
        plain = fig.axes[0]
        assert titles[0] == "Image 0 (cat)"
        assert (
            len(plain.get_images()) == 1
            and not plain.collections
            and not plain.patches
            and not plain.lines
        )
        assert np.allclose(
            plain.get_images()[0].get_array(), DS[0]["img"].numpy().transpose(1, 2, 0)
        )
        assert titles[1].startswith("Before edits: mean of 2 cat prototypes")
        assert titles[1].endswith(
            "\nPAM 0.500"
        )  # "PAM", whatever kind of mask the data has
        assert titles[2].startswith("After edits: mean of 1 cat prototypes")
        assert titles[2].endswith("\nPAM 1.000")
        assert titles[3].startswith("Change from editing")
        assert "image 0 (cat): PAM 0.500 -> 1.000" in capsys.readouterr().out
        path = temp_dir / "before_after.png"
        fig.savefig(path)
        Image.open(path).verify()
    finally:
        plt.close(fig)


class RecordsMode(ChannelsAsActivations):
    """Records whether it was in training mode each time it ran."""

    def __init__(self, weight):
        super().__init__(weight)
        self.modes = []

    def forward(self, img, return_prototype_layer_output_dict=False):
        self.modes.append(self.training)
        return super().forward(img)


def _check_figures_run_the_model_in_eval_mode_and_put_its_mode_back():
    before, after = RecordsMode(W_BEFORE).train(), RecordsMode(W_AFTER).train()
    pam_gains(before, after, DS, device="cpu")
    assert before.modes and not any(before.modes) and not any(after.modes)
    assert before.training and after.training


def _check_no_device_means_the_models_own_device_not_cuda_whenever_available(
    monkeypatch,
):
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: True
    )  # the model is on the CPU all the same
    assert pam_gains(BEFORE, AFTER, DS) == [(0.5, 0, 0.5, 1.0), (-0.5, 1, 0.5, 0.0)]


def test_figures_use_eval_mode_and_the_models_own_device(monkeypatch):
    """The model runs in eval mode (its mode is put back), on its own device when none is given."""
    _check_figures_run_the_model_in_eval_mode_and_put_its_mode_back()
    _check_no_device_means_the_models_own_device_not_cuda_whenever_available(
        monkeypatch
    )


def test_pam_is_measured_on_the_masks_grid_when_the_mask_is_larger_than_the_image(
    temp_dir,
):
    big = (
        (~lesion_mask())
        .float()[None, None]
        .repeat_interleave(2, 2)
        .repeat_interleave(2, 3)[0]
    )  # (1, 16, 16)
    ds = Dataset(
        [
            item([INSIDE, OUTSIDE, OUTSIDE], 0, mask=big),
            item([OUTSIDE, INSIDE, OUTSIDE], 0, mask=big),
        ],
        classes=["cat", "dog"],
    )
    assert [(i, b, a) for _, i, b, a in pam_gains(BEFORE, AFTER, ds, device="cpu")] == [
        (0, pytest.approx(0.5, abs=0.1), pytest.approx(1.0, abs=0.1)),
        (1, pytest.approx(0.5, abs=0.1), pytest.approx(0.0, abs=0.1)),
    ]
    fig = plot_before_after(
        BEFORE, AFTER, ds, 0, mean=torch.zeros(3), std=torch.ones(3), device="cpu"
    )
    plt.close(fig)


# ---- prototype importance before/after


PROTOS = torch.randn(3, 4, 1, 1)


def stub(weight, prototypes):
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    layer.weight.data = weight
    return SimpleNamespace(
        prototype_prediction_head=SimpleNamespace(class_connection_layer=layer),
        prototype_layer=SimpleNamespace(prototype_tensors=prototypes),
    )


# Prototypes 0 and 1 belong to class 0, prototype 2 to class 1.
IMP_BEFORE = stub(torch.tensor([[1.0, 0.5, 0.0], [0.0, 0.0, 0.8]]), PROTOS)


# The edit raised prototype 0 to 1.2 and removed prototype 1 (weight 0).
IMP_AFTER = stub(torch.tensor([[1.2, 0.0, 0.0], [0.0, 0.0, 0.8]]), PROTOS)


NAMES = ["cat", "dog"]


def approx_rows(rows):
    return [
        (j, c, pytest.approx(wb), pytest.approx(wa), req, rem)
        for j, c, wb, wa, req, rem in rows
    ]


def test_rows_give_each_prototype_its_weight_before_and_after():
    assert prototype_importance_rows(
        IMP_BEFORE, IMP_AFTER, class_names=NAMES
    ) == approx_rows(
        [
            (0, "cat", 1.0, 1.2, False, False),
            (1, "cat", 0.5, 0.0, False, True),  # removed
            (2, "dog", 0.8, 0.8, False, False),
        ]
    )


def _check_required_prototypes_are_marked_and_keep_the_models_own_weight():
    for required in (
        [0, 2],
        {0: 1.85, 2: None},
        [(0, 1.85), (2, 0.1)],
    ):  # indices; a dict; Proto-RSet's pairs
        rows = prototype_importance_rows(
            IMP_BEFORE, IMP_AFTER, required=required, class_names=NAMES
        )
        assert rows[0][3:5] == (
            pytest.approx(1.2),
            True,
        )  # the edited model's weight, never the requested one
        assert rows[1][4] is False
        assert rows[2][3:5] == (pytest.approx(0.8), True)


def _check_a_prototype_with_no_positive_weight_takes_its_class_from_the_prototype_layer():
    before = stub(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.8]]), PROTOS
    )  # prototype 1 has no last-layer class
    before.prototype_layer.prototype_class_identity = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    rows = prototype_importance_rows(before, before, class_names=NAMES)
    assert [r[1] for r in rows] == ["cat", "dog", "dog"]


def test_importance_rows_mark_required_prototypes_and_resolve_classes():
    """Required prototypes are marked and keep the model's own weight; a prototype with no positive weight takes the layer's class."""
    _check_required_prototypes_are_marked_and_keep_the_models_own_weight()
    _check_a_prototype_with_no_positive_weight_takes_its_class_from_the_prototype_layer()


def test_a_pruned_after_model_is_matched_by_prototype_tensor():
    pruned = stub(
        torch.tensor([[1.2, 0.0], [0.0, 0.8]]), PROTOS[[0, 2]]
    )  # prototype 1 pruned away
    assert prototype_importance_rows(
        IMP_BEFORE, pruned, class_names=NAMES
    ) == approx_rows(
        [
            (0, "cat", 1.0, 1.2, False, False),
            (1, "cat", 0.5, 0.0, False, True),
            (2, "dog", 0.8, 0.8, False, False),
        ]
    )


@pytest.fixture
def proto_dir(tmp_path):
    rng = np.random.default_rng(0)
    for j in range(3):
        for kind in ("original", "overlayheatmap"):
            Image.fromarray(rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)).save(
                tmp_path / f"proto_{j}_{kind}.png"
            )
    (tmp_path / "patch_info_dict.json").write_text(
        json.dumps({str(j): [1, 4, 2, 5 + j % 2] for j in range(3)})
    )
    return tmp_path


def texts(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts]


def test_plot_draws_a_row_per_prototype_and_marks_removed_and_required(
    proto_dir, tmp_path
):
    fig = plot_prototype_importance(
        IMP_BEFORE,
        IMP_AFTER,
        proto_dir,
        required={0: 1.85},
        class_names=NAMES,
        image_inches=0.5,
        save_path=tmp_path / "importance.png",
    )
    try:
        assert len(fig.axes) == 2 + 3 * 6  # 2 header cells + 6 cells per row
        shown = texts(fig)
        assert {"Prototype 0\ncat", "Prototype 1\ncat", "Prototype 2\ndog"} <= set(
            shown
        )
        assert shown.count("Removed") == 1 and shown.count("Required") == 1
        assert (
            "1.20" in shown
            and "1.85" not in shown
            and "Importance adjusted." in shown
            and "0.00" in shown
        )
        heatmaps = fig.axes[2:][2::6]  # column 2 of each row
        assert [len(ax.lines) for ax in heatmaps] == [
            0,
            2,
            0,
        ]  # red X only on the removed one
        Image.open(tmp_path / "importance.png").verify()
    finally:
        plt.close(fig)


def test_plot_shows_prototypes_as_the_other_figures_do(proto_dir):
    """The source image with the patch box 75% transparent, then the activation blended onto the image
    (proto_<j>_overlayheatmap.png), as plot_prototype_grid and plot_image_decisions show prototypes.
    """
    fig = plot_prototype_importance(
        IMP_BEFORE, IMP_AFTER, proto_dir, class_names=NAMES, image_inches=0.5
    )
    try:
        cells = fig.axes[2:]
        for r, j in enumerate([0, 1, 2]):
            source, heat = cells[6 * r + 1], cells[6 * r + 2]
            assert np.array_equal(
                np.asarray(source.get_images()[0].get_array()),
                np.asarray(Image.open(proto_dir / f"proto_{j}_original.png")),
            )
            assert np.array_equal(
                np.asarray(heat.get_images()[0].get_array()),
                np.asarray(Image.open(proto_dir / f"proto_{j}_overlayheatmap.png")),
            )
            (box,) = source.patches
            assert box.get_alpha() == 0.25 and not box.get_fill()
            # [y0, y1, x0, x1] = [1, 4, 2, 5 + j % 2]
            assert box.get_xy() == (1.5, 0.5) and (
                box.get_width(),
                box.get_height(),
            ) == (3 + j % 2, 3)
            assert not heat.patches
    finally:
        plt.close(fig)


def test_works_with_a_real_proto_rset_edit(
    projected_ppnet, saved_projected_ppnet_path, split_loaders, toy_loader, tmp_path
):
    from protopnet.visualization import save_prototype_images_to_file
    from rashomon_sets.protorset_factory import DEFAULT_RSET_ARGS, ProtoRSetFactory

    args = {
        **DEFAULT_RSET_ARGS,
        "max_iter": 1_000,
        "rashomon_bound_multiplier": 1.1,
        "lam": 1,
        "lr_for_opt": 0.01,
    }
    factory = ProtoRSetFactory(
        split_loaders,
        saved_projected_ppnet_path,
        rashomon_set_args=args,
        device="cpu",
        analysis_save_dir=tmp_path / "rset",
    )
    before = factory.produce_protopnet_object()
    removed = [
        p
        for p in range(before.prototype_layer.num_prototypes)
        if factory.require_to_avoid_prototype(p)
    ]
    after = factory.produce_protopnet_object()
    save_prototype_images_to_file(
        before,
        toy_loader,
        tmp_path / "vis",
        (224, 224),
        torch.zeros(3),
        torch.ones(3),
        device="cpu",
    )

    rows = prototype_importance_rows(before, after, class_names=["c0", "c1"])
    assert [
        r[0] for r in rows if r[5]
    ] == removed  # exactly the removed prototypes are marked
    fig = plot_prototype_importance(
        before,
        after,
        tmp_path / "vis" / "prototypes",
        class_names=["c0", "c1"],
        image_inches=0.5,
    )
    try:
        assert texts(fig).count("Removed") == len(removed)
    finally:
        plt.close(fig)
