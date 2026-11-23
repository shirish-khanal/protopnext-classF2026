"""Figures for checking Proto-RSet edits: how an edit changed a model.

- pam_gains, print_largest_pam_gains, plot_before_after: how an edit changed the activation precision
  ("PAM") of a class's mean activation map, per image.
- prototype_importance_rows, plot_prototype_importance: each prototype's last-layer weight before and
  after an edit, with removed and required prototypes marked.

The single-model figures (plot_prototype_grid, plot_image_decisions) and the helpers shared with them
are in protopnet.visualization. Nothing is dataset-specific: the models are arguments, class names come
from `class_names`, else the data (`dataset.classes`), else class indices, and images are un-normalized
with the mean and standard deviation passed in. Like protopnext's fine-annotation loss, the
"fine_annotation" mask is not validated: use these only when fine annotation is on (fa_coef set).
"""

import json

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from protopnet.metrics import (
    annotation_mask,
    prototype_classes,
    prototype_display_classes,
)
from protopnet.utilities.general_utilities import (
    class_names_from,
    eval_mode,
    given_class_names,
)
from protopnet.visualization import (
    WEIGHTED,
    _device,
    _lesion,
    _on_grid,
    _picture,
    draw_patch_box,
    mean_map,
    pam,
)

TOP_N = 5


def maps(model, item, device=None):
    """The item's prototype activation maps on its mask's grid (as the metric upsamples them), from the model
    in eval mode."""
    device = _device(model, device)
    with torch.no_grad(), eval_mode(model):
        acts = model(
            item["img"].unsqueeze(0).to(device), return_prototype_layer_output_dict=True
        )["prototype_activations"]
    return F.interpolate(
        acts,
        size=annotation_mask(item).shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0].cpu()


def edited_classes(pc_b, pc_a, num_classes):
    """Classes whose set of prototypes differs between the two models."""
    return {
        c
        for c in range(num_classes)
        if {j for j, k in pc_b.items() if k == c}
        != {j for j, k in pc_a.items() if k == c}
    }


def pam_gains(before, after, ds, device=None, weighted=WEIGHTED):
    """(gain, image index, PAM before, PAM after) for annotated images of edited classes, largest gain first."""
    pc_b, w_b = prototype_classes(before)
    pc_a, w_a = prototype_classes(after)
    edited = edited_classes(pc_b, pc_a, w_b.shape[0])

    results = []
    for i in range(len(ds)):
        item = ds[i]
        lesion = _lesion(annotation_mask(item))
        c = int(item["target"])
        if lesion is None or c not in edited:
            continue
        pb_ = pam(
            mean_map(maps(before, item, device), pc_b, w_b, c, weighted)[0], lesion
        )
        pa_ = pam(
            mean_map(maps(after, item, device), pc_a, w_a, c, weighted)[0], lesion
        )
        results.append((pa_ - pb_, i, pb_, pa_))
    results.sort(reverse=True)
    return results


def print_largest_pam_gains(results, ds, top_n=TOP_N, class_names=None):
    """Prints the top_n gains. Class names come from `class_names`, else `ds.classes`, else indices."""
    names = given_class_names(ds, class_names)
    label = (lambda c: names[c]) if names is not None else str
    print("largest PAM gains (image, before -> after):")
    for gain, i, pb_, pa_ in results[:top_n]:
        print(
            f"  image {i:2d} ({label(int(ds[i]['target']))}): {pb_:.3f} -> {pa_:.3f}  ({gain:+.3f})"
        )


def plot_before_after(
    before,
    after,
    ds,
    index,
    mean,
    std,
    device=None,
    weighted=WEIGHTED,
    class_names=None,
    split=None,
):
    """Four panels for image `index` of `ds`: the image alone (no heatmap, outline or box), before edits,
    after edits, and the change."""
    item = ds[index]
    lesion = _lesion(annotation_mask(item))
    if lesion is None:
        raise ValueError(
            f"image {index} has an empty annotation mask, so its PAM is undefined."
        )
    pc_b, w_b = prototype_classes(before)
    pc_a, w_a = prototype_classes(after)
    names = class_names_from(ds, w_b.shape[0], class_names)
    c = int(item["target"])
    mb, protos_b = mean_map(maps(before, item, device), pc_b, w_b, c, weighted)
    ma, protos_a = mean_map(maps(after, item, device), pc_a, w_a, c, weighted)
    where = f", {split}" if split else ""
    print(
        f"image {index} ({names[c]}{where}): PAM {pam(mb, lesion):.3f} -> {pam(ma, lesion):.3f}"
    )

    vmin, vmax = float(min(mb.min(), ma.min())), float(max(mb.max(), ma.max()))
    pic, grid = _picture(item, mean, std), _on_grid(lesion.shape)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(pic)
    axes[0].set_title(f"Image {index} ({names[c]})", fontsize=9)
    for ax, m, tag, protos in (
        (axes[1], mb, "Before edits", protos_b),
        (axes[2], ma, "After edits", protos_a),
    ):
        ax.imshow(pic, extent=grid)
        ax.imshow(m.numpy(), cmap="jet", alpha=0.45, vmin=vmin, vmax=vmax)
        ax.set_title(
            f"{tag}: mean of {len(protos)} {names[c]} prototypes\nPAM {pam(m, lesion):.3f}",
            fontsize=9,
        )
    d = (ma - mb).numpy()
    lim = float(np.abs(d).max()) or 1.0
    axes[3].imshow(pic, extent=grid)
    axes[3].imshow(d, cmap="bwr", alpha=0.55, vmin=-lim, vmax=lim)
    axes[3].set_title(
        "Change from editing\nred: attention increased, blue: attention decreased",
        fontsize=9,
    )
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    return fig


# ======== PROTOTYPE IMPORTANCE BEFORE AND AFTER AN EDIT ========
IMAGE_INCHES = 3.0  # image size in inches
TEXT_SCALE = 1.4  # text scale: 1.2 = 20% larger, 0.9 = 10% smaller
REMOVED_BELOW = 1e-6  # an after-weight below this (in magnitude) means "removed"


def _weights_and_prototypes(model):
    return (
        model.prototype_prediction_head.class_connection_layer.weight.detach().cpu(),
        model.prototype_layer.prototype_tensors.detach().cpu(),
    )


def _after_weight(w_before, w_after, protos_before, protos_after, class_idx, proto_idx):
    """Weight of proto_idx in the edited model (handles both zeroed and pruned prototypes)."""
    if w_after.shape == w_before.shape:
        return w_after[class_idx, proto_idx].item()
    target_tensor = protos_before[proto_idx]
    for j_after in range(protos_after.shape[0]):
        if torch.allclose(target_tensor, protos_after[j_after], atol=1e-5):
            return w_after[class_idx, j_after].item()
    return 0.0


def prototype_importance_rows(
    before, after, prototypes=None, required=None, class_names=None, dataset=None
):
    w_before, protos_before = _weights_and_prototypes(before)
    w_after, protos_after = _weights_and_prototypes(after)
    names = class_names_from(dataset, w_before.shape[0], class_names)
    required = {r[0] if isinstance(r, (tuple, list)) else r for r in (required or ())}
    classes = prototype_display_classes(before)

    rows = []
    for j in range(w_before.shape[1]) if prototypes is None else prototypes:
        k = classes[j]
        w_bef = w_before[k, j].item()
        w_aft = _after_weight(w_before, w_after, protos_before, protos_after, k, j)
        rows.append(
            (j, names[k], w_bef, w_aft, j in required, abs(w_aft) < REMOVED_BELOW)
        )
    return rows


def draw_removed_mark(ax, text_scale=TEXT_SCALE, center=(0.83, 0.17), radius=0.17):
    """Draw a bold red X inside the bottom-right corner of the heatmap image."""
    cx, cy = center
    d = radius * 0.7  # half-length of each stroke (smaller X)
    lw = 11 * text_scale  # thicker strokes (broader X)
    for x0, y0, x1, y1 in (
        (cx - d, cy - d, cx + d, cy + d),
        (cx - d, cy + d, cx + d, cy - d),
    ):
        ax.plot(
            [x0, x1],
            [y0, y1],
            transform=ax.transAxes,
            color="#d00000",
            lw=lw,
            solid_capstyle="round",
            clip_on=True,
            zorder=10,
            path_effects=[
                pe.Stroke(linewidth=lw + 3, foreground="#001f2b"),
                pe.Normal(),
            ],
        )


def plot_prototype_importance(
    before,
    after,
    proto_dir,
    prototypes=None,
    required=None,
    class_names=None,
    dataset=None,
    image_inches=IMAGE_INCHES,
    text_scale=TEXT_SCALE,
    save_path=None,
):
    rows = prototype_importance_rows(
        before, after, prototypes, required, class_names, dataset
    )
    images = {}
    for j, *_ in rows:
        for kind in ("original", "overlayheatmap"):
            path = proto_dir / f"proto_{j}_{kind}.png"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found; render the prototypes first "
                    "(protopnet.visualization.save_prototype_images_to_file)"
                )
            images[j, kind] = path

    boxes = json.loads((proto_dir / "patch_info_dict.json").read_text())

    FS, IMG_IN = text_scale, image_inches
    col_in = [
        1.1 * FS,
        IMG_IN,
        IMG_IN,
        1.6 * FS,
        1.15 * FS,
        1.6 * FS,
    ]  # label, image, heatmap, before, arrow, after
    fig_width = sum(col_in)
    row_h = IMG_IN * 1.02
    header_h = 0.55 * FS
    fig_height = header_h + len(rows) * row_h

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=200, facecolor="white")

    gs = fig.add_gridspec(
        nrows=len(rows) + 1,
        ncols=6,
        width_ratios=col_in,
        height_ratios=[header_h / row_h] + [1.0] * len(rows),
        left=0.01,
        right=0.99,
        top=0.995,
        bottom=0.005,
        wspace=0.03,
        hspace=0.04,
    )

    # Header row
    for col, label in ((3, "Before"), (5, "After")):
        ax_hdr = fig.add_subplot(gs[0, col])
        ax_hdr.axis("off")
        ax_hdr.text(
            0.5,
            0.78,
            "Prototype Importance",
            ha="center",
            va="center",
            fontsize=12 * FS,
            fontfamily="sans-serif",
        )
        ax_hdr.text(
            0.5,
            0.15,
            label,
            ha="center",
            va="center",
            fontsize=12 * FS,
            fontfamily="sans-serif",
        )

    # Prototype rows
    for row_idx, (j, cls_name, w_bef, w_aft, is_required, is_removed) in enumerate(
        rows, start=1
    ):
        # Column 0: row label ("Prototype X \n Class")
        ax_lbl = fig.add_subplot(gs[row_idx, 0])
        ax_lbl.axis("off")
        ax_lbl.text(
            0.95,
            0.5,
            f"Prototype {j}\n{cls_name}",
            ha="right",
            va="center",
            fontsize=9.5 * FS,
            fontfamily="sans-serif",
            fontweight="normal",
        )

        # Column 1: source image with the patch box, 75% transparent
        ax_orig = fig.add_subplot(gs[row_idx, 1])
        ax_orig.imshow(Image.open(images[j, "original"]))
        draw_patch_box(ax_orig, boxes[str(j)])
        ax_orig.axis("off")

        # Column 2: activation blended onto the source image (+ red X if removed)
        ax_heat = fig.add_subplot(gs[row_idx, 2])
        ax_heat.imshow(Image.open(images[j, "overlayheatmap"]))
        ax_heat.axis("off")
        if is_removed:
            draw_removed_mark(ax_heat, FS)

        # Column 3: Importance Before
        ax_bef = fig.add_subplot(gs[row_idx, 3])
        ax_bef.axis("off")
        ax_bef.text(
            0.5,
            0.5,
            f"{w_bef:.3f}",
            ha="center",
            va="center",
            fontsize=12 * FS,
            fontfamily="sans-serif",
        )

        # Column 4: Arrow (+ "Required" or "Removed" label)
        ax_arr = fig.add_subplot(gs[row_idx, 4])
        ax_arr.axis("off")
        ax_arr.annotate(
            "",
            xy=(0.88, 0.5),
            xytext=(0.12, 0.5),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>", color="#0b5d8a", lw=1.5 * FS, mutation_scale=10 * FS
            ),
        )
        if is_removed:
            ax_arr.text(
                0.5,
                0.56,
                "Removed",
                color="#d00000",
                ha="center",
                va="bottom",
                fontsize=11.5 * FS,
                fontfamily="sans-serif",
            )
        elif is_required:
            ax_arr.text(
                0.5,
                0.56,
                "Required",
                color="#196b24",
                ha="center",
                va="bottom",
                fontsize=11.5 * FS,
                fontfamily="sans-serif",
            )

        # Column 5: Importance After (+ "Importance adjusted." subtitle)
        ax_aft = fig.add_subplot(gs[row_idx, 5])
        ax_aft.axis("off")
        if is_removed:
            ax_aft.text(
                0.5,
                0.5,
                "0.00",
                ha="center",
                va="center",
                fontsize=12 * FS,
                fontfamily="sans-serif",
            )
        elif is_required:
            ax_aft.text(
                0.5,
                0.53,
                f"{w_aft:.2f}",
                ha="center",
                va="center",
                fontsize=12 * FS,
                fontfamily="sans-serif",
            )
            ax_aft.text(
                0.5,
                0.41,
                "Importance adjusted.",
                ha="center",
                va="center",
                fontsize=8 * FS,
                style="italic",
                fontfamily="sans-serif",
            )
        else:
            ax_aft.text(
                0.5,
                0.5,
                f"{w_aft:.3f}",
                ha="center",
                va="center",
                fontsize=12 * FS,
                fontfamily="sans-serif",
            )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=200, facecolor="white")
    return fig
