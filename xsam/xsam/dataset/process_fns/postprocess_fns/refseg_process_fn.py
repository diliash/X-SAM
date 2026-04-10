from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import TensorType

from ...utils.process import sem_seg_postprocess


def refseg_postprocess_fn(
    outputs,
    image_sizes,
    scaled_sizes: Optional[List[TensorType]] = None,
    mask_threshold: float = 0.5,
    **kwargs,
) -> List[Dict]:
    # Expected shapes:
    # - class_queries_logits: [B, Q, C+1]
    # - masks_queries_logits: [B, Q, H, W]
    # Some runtime combinations return unbatched tensors for B=1, so normalize here.
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits
    if class_queries_logits.ndim == 2:
        class_queries_logits = class_queries_logits.unsqueeze(0)
    if masks_queries_logits.ndim == 3:
        masks_queries_logits = masks_queries_logits.unsqueeze(0)
    scaled_sizes = scaled_sizes if scaled_sizes is not None else image_sizes
    if image_sizes is None:
        image_sizes = []
    if scaled_sizes is None:
        scaled_sizes = image_sizes
    if isinstance(image_sizes, tuple):
        image_sizes = [image_sizes]
    if isinstance(scaled_sizes, tuple):
        scaled_sizes = [scaled_sizes]

    batch_size = class_queries_logits.shape[0]

    # Loop over items in batch size
    results: List[Dict[str, TensorType]] = []

    for i in range(batch_size):
        mask_pred = masks_queries_logits[i]
        mask_cls = class_queries_logits[i]
        if i < len(image_sizes):
            image_size = image_sizes[i]
        elif len(image_sizes) > 0:
            image_size = image_sizes[0]
        else:
            image_size = mask_pred.shape[-2:]

        if i < len(scaled_sizes):
            scaled_size = scaled_sizes[i]
        elif len(scaled_sizes) > 0:
            scaled_size = scaled_sizes[0]
        else:
            scaled_size = image_size

        mask_pred = sem_seg_postprocess(mask_pred, scaled_size, image_size[0], image_size[1])

        # the last class is __background__
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]  # [Q, num_fg_labels]

        # 255 is the ignore index
        segmentation = torch.full((image_size[0], image_size[1]), 255, dtype=torch.long, device=mask_pred.device)

        if scores.numel() == 0 or scores.shape[1] == 0:
            segments_info = {
                "id": 0,
                "label_id": 0,
                "was_fused": False,
                "score": 0.0,
            }
            results.append({"segmentation": segmentation, "segments_info": segments_info})
            continue

        num_queries, num_fg_labels = scores.shape
        flat_idx = torch.argmax(scores)
        query_idx = int((flat_idx // num_fg_labels).item())
        label_idx = int((flat_idx % num_fg_labels).item())
        top_score = scores[query_idx, label_idx]

        selected_mask_pred = mask_pred[query_idx : query_idx + 1]
        mask_prob = selected_mask_pred.sigmoid()
        segmentation[mask_prob[0] > mask_threshold] = 1

        segments_info = {
            "id": 0,
            "label_id": int(label_idx),
            "was_fused": False,
            "score": round(top_score.item(), 6),
        }

        results.append({"segmentation": segmentation, "segments_info": segments_info})
    return results
