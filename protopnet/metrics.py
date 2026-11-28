# ====== MODEL AND DATA LOADING ======
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from .utilities.general_utilities import class_names_from, eval_mode


# ======== HELPER FUNCTIONS ========
def in_bbox(loc, bbox):
    return (
        loc[0] >= bbox[0]
        and loc[0] <= bbox[1]
        and loc[1] >= bbox[2]
        and loc[1] <= bbox[3]
    )


def add_gaussian_noise(norm_img, generator, std=0.2, eps=0.25):
    """
    Adds noise to an image tensor, with constraints on the noise's magnitude.

    Parameters:
    - norm_img (torch.Tensor): Normalized image tensor.
    - std (float): Standard deviation of the Gaussian noise.
    - eps (float): Maximum absolute value of the noise elements.
    - generator (torch.Generator): Generator for random numbers, specifying device.

    Returns:
    - torch.Tensor: Perturbed image.
    """
    # Create Gaussian noise with the specified standard deviation using the provided generator
    noise = torch.zeros(norm_img.shape, device=norm_img.device).normal_(
        mean=0, std=std, generator=generator
    )
    # Constrain the noise to be within the specified epsilon limits
    noise = torch.clamp(noise, min=-eps, max=eps)
    # Add the constrained noise to the original image
    perturbed_img = norm_img + noise
    return perturbed_img


# ============ METRICS ============
class InterpMetrics(Metric):
    def __init__(
        self,
        num_classes,
        part_num,
        proto_per_class,
        img_sz=224,
        half_size=36,
        dist_sync_on_step=False,
        uncropped=True,
    ):
        """
        Base class for PartStabilityScore and PartConsistencyScore

        Parameters:
        - num_classes (int): The number of classes in the dataset.
        - part_num (int): The number of distinct parts in the dataset.
        - proto_per_class (int): The number of prototypes per class.
        - img_sz (int): Input image size to the network, default 224.
        - half_size (int): half of size of the prototype bounding box, defaults to 36.
        - dist_sync_on_step (bool): torchmetrics parameter.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.half_size = half_size
        self.proto_per_class = proto_per_class
        self.img_sz = img_sz
        self.part_num = part_num
        self.uncropped = uncropped

    def filter_proto_acts(self, proto_acts, targets):
        """
        Select correct prototype activations according to ground truth labels
        output shape is batch_size, prototypes_per_class, feature_size, feature_size
        """
        feature_size = proto_acts.shape[-1]
        proto_acts_reshaped = proto_acts.view(
            proto_acts.shape[0],
            self.num_classes,
            self.proto_per_class,
            feature_size,
            feature_size,
        )
        proto_acts_selected = proto_acts_reshaped[
            torch.arange(proto_acts_reshaped.size(0)), targets
        ]
        return proto_acts_selected, targets

    def proto2part_and_masks(
        self,
        all_proto_acts,
        all_targets,
        all_sample_parts_centroids,
        all_sample_bounding_box,
    ):
        all_proto_acts = torch.cat(all_proto_acts, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_sample_bounding_box = torch.cat(all_sample_bounding_box, dim=0)

        # Enumerate all the classes, thus enumerate all the prototypes
        all_proto_to_part, all_proto_part_mask = [], []
        # `all_proto_to_part` each element indicates the corresponding object parts of a prototype on the images.
        # `all_proto_part_mask`each element indicates the existing (non-masked) object parts on the images of a prototype.
        for test_image_label in range(self.num_classes):
            arr_ids = torch.argwhere(all_targets == test_image_label).flatten().tolist()
            if len(arr_ids) == 0:
                continue
            class_proto_acts = all_proto_acts[arr_ids]
            selected_parts_centroids = [all_sample_parts_centroids[i] for i in arr_ids]
            selected_bboxes = all_sample_bounding_box[arr_ids]

            # Get part annotations on all the images of current class
            class_part_labels, class_part_masks = [], []
            """
            `class_part_labels` save the part labels of images in this class.
            `class_part_masks` save the part masks of images in this class.
            """
            for idx in range(len(class_proto_acts)):
                # Get part annotations
                part_labels, part_mask = [], torch.zeros((self.part_num,))
                bbox = selected_bboxes[idx]
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                parts_centroids = selected_parts_centroids[idx]
                for part_centroid in parts_centroids:
                    part_id = int(
                        (part_centroid[0] - 1).item()
                    )  # The id of current object part (begin from 0)
                    part_mask[
                        part_id
                    ] = 1  # The current object part exists in current image
                    if self.uncropped:
                        part_labels.append(
                            [
                                part_id,
                                int(part_centroid[1] * self.img_sz),
                                int(part_centroid[2] * self.img_sz),
                            ]
                        )
                    else:
                        centroid_x, centroid_y = (
                            part_centroid[1] - bbox_x1,
                            part_centroid[2] - bbox_y1,
                        )
                        ratio_x, ratio_y = centroid_x / (
                            bbox_x2 - bbox_x1
                        ), centroid_y / (
                            bbox_y2 - bbox_y1
                        )  # Fit the bounding boxes' coordinates to the cropped images
                        re_centroid_x, re_centroid_y = int(self.img_sz * ratio_x), int(
                            self.img_sz * ratio_y
                        )
                        part_labels.append([part_id, re_centroid_x, re_centroid_y])

                class_part_labels.append(part_labels)
                class_part_masks.append(part_mask)

            for proto_idx in range(self.proto_per_class):
                img_num = len(class_proto_acts)
                proto_to_part = torch.zeros(
                    (img_num, self.part_num)
                )  # Element = 1 -> the prototype corresponds to this object part on this image, element = 0 otherwise
                for img_idx in range(img_num):
                    part_labels = class_part_labels[
                        img_idx
                    ]  # Get the part labels of current image
                    activation_map = class_proto_acts[
                        img_idx, proto_idx
                    ]  # Get the activation map of current prototype on current image
                    activation_map = activation_map.unsqueeze(0).unsqueeze(0)
                    upsampled_activation_map = (
                        transforms.Resize(
                            size=(self.img_sz, self.img_sz),
                            interpolation=transforms.InterpolationMode.BICUBIC,
                        )(activation_map)
                        .squeeze(0)
                        .squeeze(0)
                    )

                    max_value = upsampled_activation_map.max()
                    max_indices = (upsampled_activation_map == max_value).nonzero(
                        as_tuple=True
                    )

                    # to deal with a very unlikely (unless artificial) special case
                    # where max is not unique
                    if len(max_indices[0]) > 1 and len(max_indices[1]) > 1:
                        max_indices = (max_indices[0][:1], max_indices[0][:1])

                    region_pred = (
                        max(0, max_indices[0] - self.half_size),
                        min(self.img_sz, max_indices[0] + self.half_size),
                        max(0, max_indices[1] - self.half_size),
                        min(self.img_sz, max_indices[1] + self.half_size),
                    )  # Get the corresponding region of current prototype, (y1, y2, x1, x2)
                    # Get the corresponding object parts of current prototype
                    for part_label in part_labels:
                        part_id, centroid_x_gt, centroid_y_gt = (
                            part_label[0],
                            part_label[1],
                            part_label[2],
                        )
                        if in_bbox((centroid_y_gt, centroid_x_gt), region_pred):
                            proto_to_part[img_idx, part_id] = 1

                # class_part_masks = torch.stack(class_part_masks)
                all_proto_to_part.append(proto_to_part)
                all_proto_part_mask.append(class_part_masks)

        all_proto_part_mask = [torch.stack(o) for o in all_proto_part_mask]
        return all_proto_to_part, all_proto_part_mask


class PartConsistencyScore(InterpMetrics):
    def __init__(
        self,
        num_classes,
        part_num,
        proto_per_class,
        img_sz=224,
        half_size=36,
        part_thresh=0.8,
        dist_sync_on_step=False,
        uncropped=True,
    ):
        super().__init__(
            num_classes=num_classes,
            part_num=part_num,
            proto_per_class=proto_per_class,
            img_sz=img_sz,
            half_size=half_size,
            dist_sync_on_step=dist_sync_on_step,
            uncropped=uncropped,
        )
        self.part_thresh = part_thresh
        self.add_state("all_proto_acts", default=[], dist_reduce_fx="cat")
        self.add_state("all_targets", default=[], dist_reduce_fx="cat")
        self.add_state("all_sample_parts_centroids", default=[], dist_reduce_fx="cat")
        self.add_state("all_sample_bounding_box", default=[], dist_reduce_fx="cat")

    def update(self, proto_acts, targets, sample_parts_centroids, sample_bounding_box):
        batch_proto_acts, batch_targets = self.filter_proto_acts(proto_acts, targets)
        self.all_proto_acts.append(batch_proto_acts)
        self.all_targets.append(batch_targets)
        self.all_sample_parts_centroids.extend(sample_parts_centroids)
        self.all_sample_bounding_box.append(sample_bounding_box)

    def compute(self):
        all_proto_to_part, all_proto_part_mask = self.proto2part_and_masks(
            self.all_proto_acts,
            self.all_targets,
            self.all_sample_parts_centroids,
            self.all_sample_bounding_box,
        )
        all_proto_consis = []
        # Enumerate all the prototypes to calculate consistency score
        for proto_idx in range(len(all_proto_to_part)):
            proto_to_part = all_proto_to_part[proto_idx]
            proto_part_mask = all_proto_part_mask[proto_idx]
            assert (
                (1.0 - proto_part_mask) * proto_to_part
            ).sum() == 0  # Assert that the prototype does not correspond to an object part that cannot be visualized (not in the part annotations)
            proto_to_part_sum = proto_to_part.sum(axis=0)
            proto_part_mask_sum = proto_part_mask.sum(axis=0)
            proto_part_mask_sum = torch.where(
                proto_part_mask_sum == 0, proto_part_mask_sum + 1, proto_part_mask_sum
            )  # Eliminate the 0 elements in all_part_mask_sum~(prevent 0 from being denominator), it doesn't affect the evaluation result
            mean_part_float = proto_to_part_sum / proto_part_mask_sum
            mean_part = (
                mean_part_float >= self.part_thresh
            ).float()  # The prototope is determined to be non-consistent if  no element in the averaged corresponding object parts exceeds `part_thresh`

            if mean_part.sum() == 0:
                all_proto_consis.append(0.0)
            else:
                all_proto_consis.append(1.0)

        all_proto_consis = torch.tensor(all_proto_consis).float()
        consistency_score = all_proto_consis.mean()

        return consistency_score


class PartStabilityScore(InterpMetrics):
    def __init__(
        self,
        num_classes,
        part_num,
        proto_per_class,
        img_sz=224,
        half_size=36,
        dist_sync_on_step=False,
        uncropped=True,
    ):
        super().__init__(
            num_classes=num_classes,
            part_num=part_num,
            proto_per_class=proto_per_class,
            img_sz=img_sz,
            half_size=half_size,
            dist_sync_on_step=dist_sync_on_step,
            uncropped=uncropped,
        )

        self.add_state("all_proto_acts", default=[], dist_reduce_fx="cat")
        self.add_state("all_proto_acts_noisy", default=[], dist_reduce_fx="cat")
        self.add_state("all_targets", default=[], dist_reduce_fx="cat")
        self.add_state("all_sample_parts_centroids", default=[], dist_reduce_fx="cat")
        self.add_state("all_sample_bounding_box", default=[], dist_reduce_fx="cat")

    def update(
        self,
        proto_acts,
        proto_acts_noisy,
        targets,
        sample_parts_centroids,
        sample_bounding_box,
    ):
        batch_proto_acts, batch_targets = self.filter_proto_acts(proto_acts, targets)
        self.all_proto_acts.append(batch_proto_acts)
        self.all_targets.append(batch_targets)
        self.all_sample_parts_centroids.extend(sample_parts_centroids)
        self.all_sample_bounding_box.append(sample_bounding_box)

        batch_proto_acts_noisy, _ = self.filter_proto_acts(proto_acts_noisy, targets)
        self.all_proto_acts_noisy.append(batch_proto_acts_noisy)

    def compute(self):
        all_proto_to_part, _ = self.proto2part_and_masks(
            self.all_proto_acts,
            self.all_targets,
            self.all_sample_parts_centroids,
            self.all_sample_bounding_box,
        )
        all_proto_to_part_noisy, _ = self.proto2part_and_masks(
            self.all_proto_acts_noisy,
            self.all_targets,
            self.all_sample_parts_centroids,
            self.all_sample_bounding_box,
        )

        all_proto_stability = []
        for proto_idx in range(len(all_proto_to_part)):
            proto_to_part = all_proto_to_part[proto_idx]
            proto_to_part_noise = all_proto_to_part_noisy[proto_idx]
            # Determine whether the elements in `proto_to_part` and `proto_to_part_perturb` are equal
            difference_sum = torch.abs(proto_to_part - proto_to_part_noise).sum(dim=-1)
            is_equal = difference_sum == 0
            is_equal = is_equal.float()
            proto_stability = (
                is_equal.mean()
            )  # The ratio of elements that keep unchanged under perturbation
            all_proto_stability.append(proto_stability)

        all_proto_stability = torch.tensor(all_proto_stability).float()
        stability_score = all_proto_stability.mean()

        return stability_score


# ======== ACTIVATION PRECISION ========
PERCENTILE = 0.95
ANNOTATION_KEY = "fine_annotation"


def prototype_classes_from_weight(weight):
    """
    Prototype j's class: the argmax of its last-layer column if that column's max is > 0,
    else None (it has no class, e.g. a prototype removed by a Proto-RSet edit).

    Parameters:
    - weight (torch.Tensor): The last-layer weight, shape (num_classes, num_prototypes).

    Returns: ({prototype index: class index or None}, the weight detached on the CPU).
    """
    w = weight.detach().cpu()  # (classes, prototypes)
    return {
        j: (int(w[:, j].argmax()) if w[:, j].max() > 0 else None)
        for j in range(w.shape[1])
    }, w


def prototype_classes(model):
    """
    The model's prototype classes, from its last layer (see prototype_classes_from_weight).
    This is the one place that decides where a model's prototype classes come from.

    Parameters:
    - model (ProtoPNet): A model with a linear class connection layer.

    Returns: ({prototype index: class index or None}, the last-layer weight on the CPU).
    """
    return prototype_classes_from_weight(
        model.prototype_prediction_head.class_connection_layer.weight
    )


def prototype_display_classes(model):
    """
    Each prototype's class, for showing or grouping prototypes: its last-layer class (prototype_classes,
    the class activation precision uses) and, for a prototype without one (e.g. removed by a Proto-RSet
    edit), the prototype layer's class identity.

    Parameters:
    - model (ProtoPNet): A model whose prototype layer has a prototype_class_identity.

    Returns: list of class indices, one per prototype.
    """
    classes, _ = prototype_classes(model)
    if any(c is None for c in classes.values()):
        identity = model.prototype_layer.prototype_class_identity.argmax(dim=1).tolist()
        classes = {j: identity[j] if c is None else c for j, c in classes.items()}
    return [classes[j] for j in range(len(classes))]


def annotation_mask(sample):
    """
    The batch's or item's "fine_annotation" mask, as a dense tensor. Not validated, like
    protopnext's fine-annotation loss: use it only when fine annotation is on (fa_coef set).

    Parameters:
    - sample (dict): A batch or a single item with a "fine_annotation" tensor (sparse or dense).
    """
    mask = sample[ANNOTATION_KEY]
    return mask.to_dense() if mask.is_sparse else mask


class ActivationPrecisionScore(Metric):
    """
    Activation precision, from IAIA-BL (Barnett et al., 2021, https://arxiv.org/abs/2103.12308).
    For every (annotated image, prototype of the image's class) pair, it is
    the share of the top (1 - percentile) of the prototype's activation map, upsampled to the mask's
    size, that falls on the annotated region (fine_annotation == 0). Prototype classes come from the
    last layer (see prototype_classes_from_weight). Images whose mask is all zeros are not annotated
    and are skipped, as in protopnext's fine-annotation loss.

    compute() returns the model's activation_precision_score (mean over pairs; nan if none).
    per_class() and per_prototype() give the same pairs broken down by class or by prototype.
    """

    def __init__(self, percentile=PERCENTILE, dist_sync_on_step=False):
        """
        Parameters:
        - percentile (float): Activations at or above this quantile of a map count as its top part,
            default 0.95 (the top 5%).
        - dist_sync_on_step (bool): torchmetrics parameter.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.percentile = percentile
        self.add_state("pair_classes", default=[], dist_reduce_fx="cat")
        self.add_state("pair_prototypes", default=[], dist_reduce_fx="cat")
        self.add_state("pair_precisions", default=[], dist_reduce_fx="cat")
        self._proto_class, self._weight = {}, None

    def update(self, proto_acts, targets, fine_annotation, class_connection_weight):
        """proto_acts (B, P, h, w); targets (B,); fine_annotation (B, 1, H, W); class_connection_weight (classes, P)."""
        self._proto_class, self._weight = prototype_classes_from_weight(
            class_connection_weight
        )
        mask = (
            fine_annotation.to_dense() if fine_annotation.is_sparse else fine_annotation
        )
        mask = mask.to(proto_acts.device)
        has_mask = mask.flatten(1).sum(dim=1) > 0  # all-zero mask = not annotated
        # (B, 1, H, W), True = the relevant region (e.g. a lesion), where fine annotation allows activation
        relevant = (1 - mask) > 0.5
        classes, protos, precisions = [], [], []
        for i in torch.nonzero(has_mask).flatten().tolist():
            c = int(targets[i])
            js = [
                j for j, k in self._proto_class.items() if k == c
            ]  # prototype class == image class
            if not js:
                continue
            # upsample only this image's class prototypes (same values, far less memory)
            acts = F.interpolate(
                proto_acts[i : i + 1, js],
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0]
            on_relevant = relevant[i, 0].flatten()
            for j, act in zip(js, acts):
                a = act.flatten()
                top = a >= torch.quantile(a, self.percentile)
                precisions.append(
                    (top & on_relevant).sum().item() / max(top.sum().item(), 1)
                )
                classes.append(c)
                protos.append(j)
        if precisions:
            self.pair_classes.append(torch.tensor(classes, device=self.device))
            self.pair_prototypes.append(torch.tensor(protos, device=self.device))
            self.pair_precisions.append(
                torch.tensor(precisions, dtype=torch.float, device=self.device)
            )

    def _pairs(self):
        if isinstance(self.pair_precisions, list) and not self.pair_precisions:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty, torch.empty(0)
        return (
            dim_zero_cat(self.pair_classes).cpu(),
            dim_zero_cat(self.pair_prototypes).cpu(),
            dim_zero_cat(self.pair_precisions).cpu(),
        )

    def compute(self):
        """The mean precision over all pairs, or nan if there are none."""
        _, _, precisions = self._pairs()
        return precisions.mean() if len(precisions) else torch.tensor(float("nan"))

    def per_class(self, num_classes=None):
        """{class index: (mean precision, number of pairs)} for every class of the model."""
        classes, _, precisions = self._pairs()
        if num_classes is None:
            num_classes = self._weight.shape[0] if self._weight is not None else 0
        result = {}
        for c in range(num_classes):
            p = precisions[classes == c]
            result[c] = (p.mean().item() if len(p) else float("nan"), len(p))
        return result

    def per_prototype(self):
        """{prototype index: (class index, last-layer coefficient, mean precision, number of pairs)}."""
        _, protos, precisions = self._pairs()
        result = {}
        for j in sorted(set(protos.tolist())):
            p = precisions[protos == j]
            c = self._proto_class[j]
            result[j] = (c, float(self._weight[c, j]), p.mean().item(), len(p))
        return result


def activation_precision_score(
    model, dataloader, device=None, by=None, class_names=None
):
    """
    Activation precision of `model` on `dataloader` (batches with "img", "target", "fine_annotation"),
    with the model in eval mode (its mode is put back afterwards). `device` defaults to the model's own.

    by=None: the model's score (float). by="class": {class name: (mean, n)} for every class.
    by="prototype": {prototype: (class name, coef, mean, n)}. Class names come from `class_names`,
    else the data (`dataloader.dataset.classes`), else class indices. Like protopnext's
    fine-annotation loss it does not validate the data: use it only when fine annotation is on.

    Parameters:
    - model (ProtoPNet): The model to score.
    - dataloader (torch.utils.data.DataLoader): Batches with "img", "target" and "fine_annotation".
    - device (torch.device or str, optional): Where to run the model; defaults to the model's own.
    - by (str, optional): None, "class" or "prototype", as above.
    - class_names (sequence of str, optional): Names for the class indices.
    """
    if by not in (None, "class", "prototype"):
        raise ValueError(f'by must be None, "class" or "prototype", not {by!r}')
    weight = model.prototype_prediction_head.class_connection_layer.weight
    device = next(model.parameters()).device if device is None else device
    metric = ActivationPrecisionScore().to(device)
    updated = False
    with torch.no_grad(), eval_mode(model):
        for batch in dataloader:
            mask = annotation_mask(batch)
            if not (mask.flatten(1).sum(dim=1) > 0).any():
                continue  # nothing annotated: skip the model
            acts = model(
                batch["img"].to(device), return_prototype_layer_output_dict=True
            )["prototype_activations"]
            metric.update(acts, batch["target"], mask, weight)
            updated = True

    if by is None:
        return metric.compute().item() if updated else float("nan")
    names = class_names_from(
        getattr(dataloader, "dataset", None), weight.shape[0], class_names
    )
    if by == "class":
        return {
            names[c]: v
            for c, v in metric.per_class(num_classes=weight.shape[0]).items()
        }
    elif by == "prototype":
        return {
            j: (names[c], coef, m, n)
            for j, (c, coef, m, n) in metric.per_prototype().items()
        }
