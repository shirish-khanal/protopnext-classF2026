"""Tests for protopnet/visualization.py: the figures for annotated images and the options added to it.

1. PAM, the decision figure and the prototype grid: stub models use the image's channels as their
   prototype activation maps, so every expected value can be worked out by hand.
2. The options (rendering chosen prototypes or one class, analysing one chosen image, ranking prototypes
   by last-layer weight or logit, explaining the c-th class, returned paths), on a small real model.

Nothing is dataset-specific: class names come from the data (or `class_names`), else class indices.
The before/after-edit figures are tested with Proto-RSet, in test_protorset.py.
"""

import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import protopnet.visualization as viz  # noqa: E402
from protopnet.activations import CosPrototypeActivation  # noqa: E402
from protopnet.backbones import construct_backbone  # noqa: E402
from protopnet.embedding import AddonLayers  # noqa: E402
from protopnet.models.vanilla_protopnet import VanillaProtoPNet  # noqa: E402
from protopnet.utilities.project_utilities import hash_func  # noqa: E402
from protopnet.visualization import (  # noqa: E402
    mean_map,
    pam,
    plot_image_decisions,
    plot_prototype_grid,
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
    TensorDatasetDict,
    W,
    item,
    lesion_mask,
)

# ---- PAM and the class mean map


def _check_pam_is_share_of_top_activation_on_lesion():
    assert pam(INSIDE, lesion_mask()) == 1.0
    assert pam(OUTSIDE, lesion_mask()) == 0.0


def _check_mean_map_weights_by_class_coefficients():
    acts = torch.stack(
        [torch.full((H, W), 1.0), torch.full((H, W), 3.0), torch.full((H, W), 100.0)]
    )
    pc = {0: 0, 1: 0, 2: 1}
    w = torch.tensor([[1.0, 3.0, 0.0], [0.0, 0.0, 1.0]])

    weighted, protos = mean_map(acts, pc, w, 0, weighted=True)
    unweighted, _ = mean_map(acts, pc, w, 0, weighted=False)

    assert protos == [0, 1]
    assert torch.allclose(weighted, torch.full((H, W), 2.5))  # (1*1 + 3*3) / (1 + 3)
    assert torch.allclose(unweighted, torch.full((H, W), 2.0))


def test_pam_and_the_class_mean_map():
    """PAM is the share of the top activation on the lesion; the mean map weights by class coefficients."""
    _check_pam_is_share_of_top_activation_on_lesion()
    _check_mean_map_weights_by_class_coefficients()


# ---- each annotated image: the prediction's activation and the prototypes behind it


def source_patches(folder, n):
    """proto_<j>_overlayheatmap.png for j < n (as save_prototype_images_to_file names them),
    each a different colour."""
    for j in range(n):
        Image.fromarray(np.full((4, 4, 3), 250 - 20 * j, dtype=np.uint8)).save(
            folder / f"proto_{j}_overlayheatmap.png"
        )
    return folder


class Classifier(ChannelsAsActivations):
    """Also classifies as protopnext's prediction head does (k_for_topk=1): each prototype's similarity is its
    largest activation, and the logits are the last layer applied to the similarities.
    """

    def forward(
        self,
        img,
        return_prototype_layer_output_dict=False,
        return_similarity_score_to_each_prototype=False,
    ):
        similarities = img.amax(dim=(2, 3))
        return {
            "prototype_activations": img,
            "similarity_score_to_each_prototype": similarities,
            "logits": self.prototype_prediction_head.class_connection_layer(
                similarities
            ),
        }


def decisions(model, ds, folder, **kwargs):
    return plot_image_decisions(
        model,
        ds,
        folder,
        mean=torch.zeros(3),
        std=torch.ones(3),
        device="cpu",
        **kwargs,
    )


def approx_decisions(d):
    return {
        i: (
            pred,
            pytest.approx(p),
            [(j, pytest.approx(s), pytest.approx(w)) for j, s, w in top],
        )
        for i, (pred, p, top) in d.items()
    }


def test_decisions_rank_prototypes_by_weight_times_similarity(temp_dir):
    # similarities 0.2, 0.9, 0.5; class-0 weights 1, 0.5, 2: contributions 0.2, 0.45, 1.0
    model = Classifier(torch.tensor([[1.0, 0.5, 2.0], [0.0, 0.0, 0.1]]))
    ds = Dataset([item([0.2 * INSIDE, 0.9 * INSIDE, 0.5 * INSIDE], 0)])
    fig, d = decisions(model, ds, source_patches(temp_dir, 3), top_k=2)
    plt.close(fig)
    assert d == approx_decisions({0: (0, 1.0, [(2, 0.5, 2.0), (1, 0.9, 0.5)])})


def test_decisions_use_the_predicted_class(temp_dir):
    # every similarity is 1, so logits are (2, 1): image 2 (a dog) is predicted cat
    fig, d = decisions(Classifier(W_BEFORE), DS, source_patches(temp_dir, 3))
    plt.close(fig)
    # PAM of the cat map: (INSIDE + OUTSIDE) / 2 for images 0 and 1, INSIDE alone for image 2's channel 0 and 1
    assert d == approx_decisions(
        {
            0: (0, 0.5, [(0, 1.0, 1.0), (1, 1.0, 1.0)]),
            1: (0, 0.5, [(0, 1.0, 1.0), (1, 1.0, 1.0)]),
            2: (0, 1.0, [(0, 1.0, 1.0), (1, 1.0, 1.0)]),
        }
    )


def test_decisions_show_only_prototypes_that_add_to_the_prediction(temp_dir):
    # after the edit, prototype 1's weight is 0, so only prototype 0 adds to the cat logit
    ds = Dataset([item([INSIDE, INSIDE, 0.5 * OUTSIDE], 0)])
    fig, d = decisions(Classifier(W_AFTER), ds, source_patches(temp_dir, 3))
    try:
        assert d == approx_decisions({0: (0, 1.0, [(0, 1.0, 1.0)])})
        assert len(fig.axes) == 3
    finally:
        plt.close(fig)


def test_decisions_draw_the_lesion_outline_50_percent_transparent(temp_dir):
    fig, _ = decisions(Classifier(W_BEFORE), DS, source_patches(temp_dir, 3))
    try:
        lime = matplotlib.colors.to_rgba("lime")[:3]
        for ax in (
            np.array(fig.axes).reshape(3, -1)[:, :2].flat
        ):  # image and activation columns
            outlines = [
                c for c in ax.collections if tuple(c.get_edgecolor()[0][:3]) == lime
            ]
            others = [
                c for c in ax.collections if tuple(c.get_edgecolor()[0][:3]) != lime
            ]
            assert outlines and all(c.get_alpha() == 0.5 for c in outlines)
            assert all(
                c.get_alpha() is None for c in others
            )  # the white dashed top 5% stays opaque
    finally:
        plt.close(fig)


def test_decisions_show_one_row_per_image(temp_dir, capsys):
    fig, _ = decisions(
        Classifier(W_BEFORE), DS, source_patches(temp_dir, 3), split="test"
    )
    try:
        grid = np.array(fig.axes).reshape(3, 4)
        for r in range(3):
            shown = np.asarray(grid[r, 0].get_images()[0].get_array())
            assert np.array_equal(
                shown, ITEMS[r]["img"].numpy().transpose(1, 2, 0)
            )  # the plain image
            assert (
                len(grid[r, 0].get_images()) == 1 and grid[r, 0].collections
            )  # with its annotation outlined
            assert (
                len(grid[r, 1].get_images()) == 2
            )  # the image, then the predicted class's activation over it
            for col, j in (
                (2, 0),
                (3, 1),
            ):  # the prototypes' activation heatmaps from the prototype grid
                shown = np.asarray(grid[r, col].get_images()[0].get_array())
                assert np.array_equal(
                    shown,
                    np.asarray(Image.open(temp_dir / f"proto_{j}_overlayheatmap.png")),
                )
        assert [[ax.get_title() for ax in row] for row in grid] == [
            [
                "image 0 (cat)\nannotation",
                "predicted cat\nPAM 0.50",
                "prototype 0 (cat)\n1.00 × 1.00 = 1.00",
                "prototype 1 (cat)\n1.00 × 1.00 = 1.00",
            ],
            [
                "image 1 (cat)\nannotation",
                "predicted cat\nPAM 0.50",
                "prototype 0 (cat)\n1.00 × 1.00 = 1.00",
                "prototype 1 (cat)\n1.00 × 1.00 = 1.00",
            ],
            [
                "image 2 (dog)\nannotation",
                "predicted cat (wrong)\nPAM 1.00",
                "prototype 0 (cat)\n1.00 × 1.00 = 1.00",
                "prototype 1 (cat)\n1.00 × 1.00 = 1.00",
            ],
        ]
        assert fig._suptitle.get_text().startswith("test\nlesion outline (green)")
        assert (
            "3 annotated images (3 in the data), 2 predicted correctly; mean PAM 0.667"
            in capsys.readouterr().out
        )
        fig.savefig(temp_dir / "decisions.png")
        Image.open(temp_dir / "decisions.png").verify()
    finally:
        plt.close(fig)


def test_decisions_group_rows_by_class(temp_dir):
    fig, _ = decisions(
        Classifier(W_BEFORE),
        Dataset([ITEMS[2], ITEMS[0], ITEMS[1]], classes=["cat", "dog"]),
        source_patches(temp_dir, 3),
    )
    try:
        assert [ax.get_title() for ax in np.array(fig.axes).reshape(3, 4)[:, 0]] == [
            "image 1 (cat)\nannotation",
            "image 2 (cat)\nannotation",
            "image 0 (dog)\nannotation",
        ]
    finally:
        plt.close(fig)


def test_decisions_skip_unannotated_images_and_stop_at_max_images(temp_dir):
    unannotated = item([INSIDE, OUTSIDE, OUTSIDE], 0, mask=torch.zeros(1, H, W))
    ds = Dataset([unannotated, *ITEMS], classes=["cat", "dog"])
    fig, d = decisions(
        Classifier(W_BEFORE), ds, source_patches(temp_dir, 3), max_images=1
    )
    plt.close(fig)
    assert sorted(d) == [
        1,
        3,
    ]  # image 0 is not annotated; class 0 keeps only its first annotated image


def test_decisions_need_an_annotated_image(temp_dir):
    ds = Dataset([item([INSIDE, OUTSIDE, OUTSIDE], 0, mask=torch.zeros(1, H, W))])
    with pytest.raises(ValueError, match="no annotated image"):
        decisions(Classifier(W_BEFORE), ds, source_patches(temp_dir, 3))


def _check_decisions_leave_a_gap_before_the_prototypes(temp_dir):
    fig, _ = decisions(Classifier(W_BEFORE), DS, source_patches(temp_dir, 3))
    try:
        fig.canvas.draw()
        row = [ax.get_position() for ax in np.array(fig.axes).reshape(3, 4)[0]]
        image_to_prototypes = (
            row[2].x0 - row[1].x1
        )  # between the decision map and the first prototype
        prototype_to_prototype = row[3].x0 - row[2].x1
        assert image_to_prototypes > 2 * prototype_to_prototype, (
            image_to_prototypes,
            prototype_to_prototype,
        )
    finally:
        plt.close(fig)


def _check_decisions_title_fits_the_figure_width(temp_dir):
    fig, _ = decisions(
        Classifier(W_BEFORE),
        DS,
        source_patches(temp_dir, 3),
        split="fine (finer masks, test_finer)",
    )
    try:
        fig.canvas.draw()
        title = fig._suptitle.get_window_extent(fig.canvas.get_renderer())
        assert 0 <= title.x0 and title.x1 <= fig.bbox.width, (
            title.x0,
            title.x1,
            fig.bbox.width,
        )
    finally:
        plt.close(fig)


def _check_decisions_title_clears_the_top_row_of_a_tall_figure(temp_dir):
    fig, _ = decisions(
        Classifier(W_BEFORE), Dataset([ITEMS[0]] * 9), source_patches(temp_dir, 3)
    )  # 9 rows
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title_bottom = fig._suptitle.get_window_extent(renderer).y0
        top_row = max(ax.title.get_window_extent(renderer).y1 for ax in fig.axes[:4])
        assert top_row <= title_bottom
    finally:
        plt.close(fig)


def test_decisions_layout(temp_dir):
    """A gap before the prototypes; the title fits the width and clears the top row."""
    _check_decisions_leave_a_gap_before_the_prototypes(temp_dir)
    _check_decisions_title_fits_the_figure_width(temp_dir)
    _check_decisions_title_clears_the_top_row_of_a_tall_figure(temp_dir)


# ---- the prototype grid: each class's prototypes, with their weight and activation precision


class WithPrototypeClasses(ChannelsAsActivations):
    """Also carries the prototype layer's class identity (prototypes x classes), as a ProtoPNet does."""

    def __init__(self, weight, proto_classes):
        super().__init__(weight)
        identity = torch.zeros(len(proto_classes), weight.shape[0])
        identity[range(len(proto_classes)), proto_classes] = 1
        self.prototype_layer = SimpleNamespace(prototype_class_identity=identity)


def grid_images(folder, n):
    """proto_<j>_original.png and proto_<j>_overlayheatmap.png for j < n, each a different colour, and
    patch_info_dict.json with prototype j's box [y0, y1, x0, x1], as save_prototype_images_to_file writes them.
    """
    for j in range(n):
        Image.fromarray(np.full((8, 8, 3), 20 * j, dtype=np.uint8)).save(
            folder / f"proto_{j}_original.png"
        )
        Image.fromarray(np.full((8, 8, 3), 250 - 20 * j, dtype=np.uint8)).save(
            folder / f"proto_{j}_overlayheatmap.png"
        )
    (folder / "patch_info_dict.json").write_text(
        json.dumps({str(j): [1, 4 + j, 2, 6] for j in range(n)})
    )
    return folder


def shown(ax):
    return np.asarray(ax.get_images()[0].get_array())


AP_SPLITS = {"lesion (box masks, test)": torch.utils.data.DataLoader(DS, batch_size=3)}


def _check_grid_draws_one_figure_per_class_with_the_source_image_above_its_heatmap(
    temp_dir,
):
    figs = plot_prototype_grid(
        WithPrototypeClasses(W_BEFORE, [0, 0, 1]),
        grid_images(temp_dir, 3),
        class_names=["cat", "dog"],
    )
    try:
        assert [f._suptitle.get_text() for f in figs] == ["cat", "dog"]
        cat = np.array(figs[0].axes).reshape(2, 2)
        for col, j in enumerate([0, 1]):
            assert np.array_equal(
                shown(cat[0, col]),
                np.asarray(Image.open(temp_dir / f"proto_{j}_original.png")),
            )
            assert np.array_equal(
                shown(cat[1, col]),
                np.asarray(Image.open(temp_dir / f"proto_{j}_overlayheatmap.png")),
            )
        assert len(figs[1].axes) == 2  # dog: prototype 2 only
    finally:
        for f in figs:
            plt.close(f)


def _check_grid_draws_the_patch_box_75_percent_transparent_so_the_image_shows_through(
    temp_dir,
):
    figs = plot_prototype_grid(
        WithPrototypeClasses(W_BEFORE, [0, 0, 1]),
        grid_images(temp_dir, 3),
        class_names=["cat", "dog"],
    )
    try:
        for f, js in zip(figs, [[0, 1], [2]]):
            for ax, j in zip(f.axes[: len(js)], js):  # top row
                (box,) = ax.patches
                assert box.get_alpha() == 0.25 and not box.get_fill()
                assert box.get_edgecolor()[:3] == matplotlib.colors.to_rgb("yellow")
                # [y0, y1, x0, x1] = [1, 4 + j, 2, 6]: pixels x 2..5, y 1..3+j, edges on the pixel borders
                assert box.get_xy() == (1.5, 0.5) and (
                    box.get_width(),
                    box.get_height(),
                ) == (4, 3 + j)
            assert all(
                not ax.patches for ax in f.axes[len(js) :]
            )  # no box on the heatmap
    finally:
        for f in figs:
            plt.close(f)


def test_grid_layout(temp_dir):
    """One figure per class, source image above its heatmap, patch box 75% transparent."""
    _check_grid_draws_one_figure_per_class_with_the_source_image_above_its_heatmap(
        temp_dir
    )
    _check_grid_draws_the_patch_box_75_percent_transparent_so_the_image_shows_through(
        temp_dir
    )


def _check_grid_titles_carry_the_weight_and_one_ap_line_per_split(temp_dir):
    splits = {
        **AP_SPLITS,
        "fine (finer masks, test_finer)": torch.utils.data.DataLoader(
            Dataset([ITEMS[2]]), batch_size=1
        ),
    }
    figs = plot_prototype_grid(
        WithPrototypeClasses(W_BEFORE, [0, 0, 1]),
        grid_images(temp_dir, 3),
        splits,
        device="cpu",
        class_names=["cat", "dog"],
    )
    try:
        titles = [[ax.get_title() for ax in f.axes if ax.get_title()] for f in figs]
        # the finer split has only a dog image, so the cat prototypes have no finer AP and it is left out
        assert titles == [
            [
                "prototype 0\nweight 1.00\nlesion AP 0.500",
                "prototype 1\nweight 1.00\nlesion AP 0.500",
            ],
            ["prototype 2\nweight 1.00\nlesion AP 1.000\nfine AP 1.000"],
        ]
    finally:
        for f in figs:
            plt.close(f)


def _check_grid_lists_a_prototype_under_its_last_layer_class_the_class_its_ap_is_for(
    temp_dir,
):
    # prototype 1's layer identity says cat, but its last-layer weight is on dog, as class_specific=False allows
    weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    figs = plot_prototype_grid(
        WithPrototypeClasses(weight, [0, 0, 1]),
        grid_images(temp_dir, 3),
        AP_SPLITS,
        device="cpu",
        class_names=["cat", "dog"],
    )
    try:
        titles = [[ax.get_title() for ax in f.axes if ax.get_title()] for f in figs]
        assert titles == [
            ["prototype 0\nweight 1.00\nlesion AP 0.500"],
            [
                "prototype 1\nweight 1.00\nlesion AP 1.000",
                "prototype 2\nweight 1.00\nlesion AP 1.000",
            ],
        ]
    finally:
        for f in figs:
            plt.close(f)


def _check_grid_keeps_a_prototype_removed_by_an_edit(temp_dir):
    figs = plot_prototype_grid(
        WithPrototypeClasses(W_AFTER, [0, 0, 1]),
        grid_images(temp_dir, 3),
        AP_SPLITS,
        device="cpu",
        class_names=["cat", "dog"],
    )
    try:
        assert [ax.get_title() for ax in figs[0].axes if ax.get_title()] == [
            "prototype 0\nweight 1.00\nlesion AP 0.500",
            "prototype 1\nweight 0.00",
        ]  # a prototype's AP ignores the others
    finally:
        for f in figs:
            plt.close(f)


def test_grid_class_placement(temp_dir):
    """A prototype is listed under its last-layer class; a removed prototype stays."""
    _check_grid_lists_a_prototype_under_its_last_layer_class_the_class_its_ap_is_for(
        temp_dir
    )
    _check_grid_keeps_a_prototype_removed_by_an_edit(temp_dir)


def _check_grid_without_ap_loaders_shows_only_the_weight_and_takes_class_names_from_nothing(
    temp_dir,
):
    figs = plot_prototype_grid(
        WithPrototypeClasses(W_BEFORE, [0, 0, 1]), grid_images(temp_dir, 3)
    )
    try:
        assert [f._suptitle.get_text() for f in figs] == ["0", "1"]
        assert figs[1].axes[0].get_title() == "prototype 2\nweight 1.00"
    finally:
        for f in figs:
            plt.close(f)


def test_grid_titles(temp_dir):
    """Titles carry the weight and one AP line per split; without loaders, only the weight."""
    _check_grid_titles_carry_the_weight_and_one_ap_line_per_split(temp_dir)
    _check_grid_without_ap_loaders_shows_only_the_weight_and_takes_class_names_from_nothing(
        temp_dir
    )


# ---- prototype importance before/after


# ---- the options added to protopnet/visualization.py, on a small real model


SIZE = (64, 64)  # gives resnet18 a 2x2 activation map, so heatmaps are not constant
MEAN, STD = torch.tensor([0.4, 0.5, 0.6]), torch.tensor([0.2, 0.25, 0.3])


@pytest.fixture
def loader():
    data = torch.randn(6, 3, *SIZE)
    labels = torch.tensor([1, 1, 1, 0, 0, 0])
    return DataLoader(TensorDatasetDict(data, labels), batch_size=3, shuffle=False)


@pytest.fixture
def model(loader):
    backbone = construct_backbone("resnet18")
    ppnet = VanillaProtoPNet(
        backbone=backbone,
        add_on_layers=AddonLayers(
            num_prototypes=4,
            input_channels=backbone.latent_dimension[0],
            proto_channels=16,
        ),
        activation=CosPrototypeActivation(),
        num_classes=2,
        num_prototypes_per_class=2,
    ).eval()
    ppnet.project(loader)
    return ppnet


def render(model, loader, save_loc, **kw):
    return viz.save_prototype_images_to_file(
        model, loader, save_loc, SIZE, MEAN, STD, device="cpu", **kw
    )


def names(paths):
    return {p[3].name for p in paths}  # the proto_bbox image of each rendered prototype


def _check_save_only_the_chosen_prototypes(model, loader, tmp_path):
    paths = render(model, loader, tmp_path, specify_proto_idx=[0, 2])
    assert names(paths) == {"proto_0_proto_bbox.png", "proto_2_proto_bbox.png"}
    assert sorted(
        p.name for p in (tmp_path / "prototypes").glob("*_proto_bbox.png")
    ) == [
        "proto_0_proto_bbox.png",
        "proto_2_proto_bbox.png",
    ]


def _check_save_only_one_class(model, loader, tmp_path):
    # VanillaProtoPNet: prototypes 0, 1 belong to class 0 and 2, 3 to class 1
    assert names(render(model, loader, tmp_path, specify_proto_class=1)) == {
        "proto_2_proto_bbox.png",
        "proto_3_proto_bbox.png",
    }


def test_render_only_chosen_prototypes_or_one_class(model, loader, tmp_path):
    """Rendering can be limited to chosen prototypes or to one class."""
    _check_save_only_the_chosen_prototypes(model, loader, tmp_path)
    _check_save_only_one_class(model, loader, tmp_path)


def test_render_rejects_chosen_prototypes_and_a_class_together(model, loader, tmp_path):
    """specify_proto_idx and specify_proto_class cannot be set together."""
    with pytest.raises(ValueError):
        render(model, loader, tmp_path, specify_proto_idx=[0], specify_proto_class=0)


def test_filtered_renders_add_to_patch_info_instead_of_replacing_it(
    model, loader, tmp_path
):
    render(model, loader, tmp_path, specify_proto_class=0)
    render(model, loader, tmp_path, specify_proto_class=1)
    patch_info = json.loads(
        (tmp_path / "prototypes" / "patch_info_dict.json").read_text()
    )
    assert sorted(patch_info) == ["0", "1", "2", "3"]


def test_local_analysis_of_one_chosen_image(model, loader, tmp_path):
    render(model, loader, tmp_path)
    paths = viz.local_analysis(
        model,
        tmp_path,
        loader,
        loader,
        SIZE,
        MEAN,
        STD,
        "cpu",
        run_proto_vis=False,
        specify_img_idx=4,
    )
    assert [p.name for p in paths] == [f"{hash_func(loader.dataset[4]['img'])}.png"]


def test_local_analysis_can_render_the_prototypes_itself(model, loader, tmp_path):
    paths = viz.local_analysis(
        model, tmp_path, loader, loader, SIZE, MEAN, STD, "cpu", run_proto_vis=True
    )
    assert len(paths) == 6
    assert len(list((tmp_path / "prototypes").glob("*_proto_bbox.png"))) == 4


@pytest.fixture
def chosen_prototypes(monkeypatch):
    """Records which prototypes local_analysis hands to the plotter, per image."""
    calls = []
    plotter = viz.local_analysis_plotter

    def spy(**kwargs):
        calls.append(list(kwargs["proto_indices"]))
        return plotter(**kwargs)

    monkeypatch.setattr(viz, "local_analysis_plotter", spy)
    return calls


def expected_top(model, loader, scores_from, k=3):
    """Top-k prototype indices per image, from scores_from(similarities, logits)."""
    out = []
    with torch.no_grad():
        for batch in loader:
            o = model(
                batch["img"],
                return_prototype_layer_output_dict=True,
                return_similarity_score_to_each_prototype=True,
            )
            scores = scores_from(o["similarity_score_to_each_prototype"], o["logits"])
            out.extend(torch.topk(scores, k=k, dim=1).indices.tolist())
    return out


COEF = torch.tensor([1.0, 0.0, 2.0, 0.5])  # prototype 1 was removed (coefficient 0)


def run_local(model, loader, tmp_path, **kw):
    render(model, loader, tmp_path)
    viz.local_analysis(
        model,
        tmp_path,
        loader,
        loader,
        SIZE,
        MEAN,
        STD,
        "cpu",
        run_proto_vis=False,
        **kw,
    )


def test_last_layer_coef_ranks_only_prototypes_with_a_positive_coefficient(
    model, loader, tmp_path, chosen_prototypes
):
    run_local(model, loader, tmp_path, last_layer_coef=COEF)
    assert chosen_prototypes == expected_top(
        model, loader, lambda sim, logits: sim * (COEF > 0)
    )


def test_sort_using_logit_ranks_by_similarity_times_coefficient(
    model, loader, tmp_path, chosen_prototypes
):
    run_local(model, loader, tmp_path, last_layer_coef=COEF, sort_using_logit=True)
    assert chosen_prototypes == expected_top(
        model, loader, lambda sim, logits: sim * COEF
    )


def test_reasoning_for_top_c_keeps_prototypes_of_the_cth_class(
    model, loader, tmp_path, chosen_prototypes
):
    identity = model.prototype_layer.prototype_class_identity

    def second_class_scores(sim, logits):
        c = torch.kthvalue(-logits, k=2, dim=1).indices
        return sim * ((COEF[None, :] * identity[:, c].T) > 0)

    run_local(model, loader, tmp_path, last_layer_coef=COEF, reasoning_for_top_c=2)
    assert chosen_prototypes == expected_top(model, loader, second_class_scores)


def test_global_analysis_returns_its_two_images(model, loader, tmp_path):
    render(model, loader, tmp_path)
    views, heatmap = viz.global_analysis(
        model, tmp_path, tmp_path / "prototypes", 0, loader, SIZE, MEAN, STD, "cpu"
    )
    assert (views.name, heatmap.name) == ("0_view_3.png", "0_view_3_heatmap.png")
    assert views.exists() and heatmap.exists()
