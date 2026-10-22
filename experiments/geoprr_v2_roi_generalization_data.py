"""Train-only source ROI expansion with crop-consistent geometry targets."""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from experiments.geoprr_v2_data import GeoPRRV2Dataset
from experiments.geoprr_v2_reference_relation import ReferenceRelationDataset


def _expanded_bbox(bbox, padding):
    bbox = np.asarray(bbox, dtype=np.float64)
    if bbox.shape != (4,) or not np.isfinite(bbox).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("source ROI padding requires a valid dial bbox")
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    left, top, right, bottom = padding
    return [bbox[0] - left * width, bbox[1] - top * height,
            bbox[2] + right * width, bbox[3] + bottom * height]


class _SourceROIPaddingDataset(GeoPRRV2Dataset):
    def __init__(self, *args, roi_policy=None, **kwargs):
        super().__init__(*args, **kwargs)
        policy = {"probability": .8, "max_padding": .3}
        policy.update(roi_policy or {})
        probability, maximum = float(policy["probability"]), float(policy["max_padding"])
        if not np.isfinite([probability, maximum]).all() or not 0 <= probability <= 1 or maximum < 0:
            raise ValueError("ROI probability must be in [0,1] and maximum padding must be nonnegative")
        self._roi_policy = {"probability": probability, "max_padding": maximum}

    @property
    def augmentation_policy(self):
        return dict(self._roi_policy)

    @contextmanager
    def source_roi(self, index):
        probability = self._roi_policy["probability"]
        if not self.augment or probability == 0 or np.random.random() >= probability:
            yield
            return
        original = self.rows[index]
        metadata = dict(original.get("metadata", {}))
        bbox = metadata.get("dial_bbox")
        padding = np.random.uniform(0, self._roi_policy["max_padding"], 4)
        expanded = _expanded_bbox(bbox, padding)
        metadata.setdefault("homography_bbox", list(bbox))
        metadata["dial_bbox"] = expanded
        row = dict(original, metadata=metadata)
        row.pop("roi_path", None)
        previous_calibration = self._calibrations.pop(index, None)
        self.rows[index] = row
        try:
            yield
        finally:
            self.rows[index] = original
            self._calibrations.pop(index, None)
            if previous_calibration is not None:
                self._calibrations[index] = previous_calibration


class ROIPaddingReferenceRelationDataset(ReferenceRelationDataset):
    """Preserve source exposures while expanding donor and recipient crops independently."""

    def __init__(self, manifest, *, data_root, data_config, augment, pair_manifest=None, roi_policy=None):
        super().__init__(manifest, data_root=data_root, data_config=data_config,
                         augment=augment, pair_manifest=pair_manifest)
        manifest_root = self.base.manifest_root
        self.base = _SourceROIPaddingDataset(self.rows, dataset_root=data_root, augment=augment,
            require_geometry=True, pair_manifest=None, roi_policy=roi_policy, **data_config)
        self.base.manifest_root = manifest_root
        self.rows = self.base.rows

    @property
    def augmentation_policy(self):
        return self.base.augmentation_policy

    def _sample(self, index, photometric):
        # The parent reads span from the crop's calibration before this context restores it.
        with self.base.source_roi(index):
            return super()._sample(index, photometric)
