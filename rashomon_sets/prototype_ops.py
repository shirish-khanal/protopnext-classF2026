"""Adding prototypes to a protopnext ProtoPNet.

Replaces ``ProtoPNet.add_additional_prototype`` from proto-rset's old protopnet
fork, which protopnext no longer has: protopnext keeps typed per-prototype
metadata (``prototype_meta``) and ``num_prototypes`` is derived from the tensors.
"""

from typing import List, Union

import torch
import torch.nn as nn

from protopnet.prototype_layers import (
    PrototypeFromSampleMeta,
    PrototypeWithMeta,
    latent_prototype_patch_map,
)
from protopnet.prototypical_part_model import ProtoPNet
from protopnet.utilities.project_utilities import hash_func


def prototype_from_latent(
    latent: torch.Tensor,
    h: int,
    w: int,
    img: torch.Tensor,
    sample_id: Union[int, str, torch.Tensor],
    target: Union[int, torch.Tensor],
) -> PrototypeWithMeta:
    """Build a 1x1 prototype from position (h, w) of one sample's embedding.

    Args:
        latent: the sample's (C, H, W) embedding (backbone + add-on layers).
        h, w: the latent patch to use.
        img: the input image the embedding came from (used for the source hash).
        sample_id: the sample's id in its dataloader.
        target: the sample's class label; the prototype belongs to this class.
    """
    h, w = int(h), int(w)
    return PrototypeWithMeta(
        prototype_tensor=latent[:, h : h + 1, w : w + 1].detach().clone(),
        meta=PrototypeFromSampleMeta(
            sample_id=sample_id,
            target=target,
            hash=hash_func(img),
            latent_patches=latent_prototype_patch_map(
                latent.shape[-2:], [(h, h + 1), (w, w + 1)], device=latent.device
            ),
        ),
    )


def append_prototypes(
    model: ProtoPNet, new_prototypes: List[PrototypeWithMeta]
) -> None:
    """Append prototypes to ``model`` in place, keeping layer and head aligned.

    Each new prototype is connected only to its own class in the last layer
    (weight 1.0 for its target class, 0.0 elsewhere). Requires every existing
    prototype to have sample metadata, i.e. the model must have been projected.
    """
    layer = model.prototype_layer
    head = model.prototype_prediction_head

    existing_meta = layer.prototype_meta
    if any(not isinstance(meta, PrototypeFromSampleMeta) for meta in existing_meta):
        raise ValueError(
            "Every existing prototype needs sample metadata to append prototypes; "
            "project the model first (model.project(dataloader))."
        )

    num_old = layer.num_prototypes
    existing = [
        PrototypeWithMeta(
            prototype_tensor=layer.prototype_tensors[i].detach(), meta=meta
        )
        for i, meta in enumerate(existing_meta)
    ]
    layer.set_prototypes(existing + list(new_prototypes))

    old_weight = head.class_connection_layer.weight.data
    class_identity = layer.prototype_class_identity
    new_columns = class_identity[num_old:].T.to(
        device=old_weight.device, dtype=old_weight.dtype
    )
    new_last_layer = nn.Linear(
        layer.num_prototypes, old_weight.shape[0], bias=False
    ).to(old_weight.device)
    new_last_layer.weight.data = torch.cat([old_weight, new_columns], dim=1)

    head.class_connection_layer = new_last_layer
    head.prototype_class_identity = class_identity.clone()
    head.num_prototypes = layer.num_prototypes
