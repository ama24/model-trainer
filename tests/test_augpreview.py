import numpy as np

from model_trainer import augpreview
from model_trainer.datasets import formats


def test_mask_polygons_overlap_and_separate():
    overlap = np.zeros((1, 40, 80), np.uint8)
    overlap[0, 5:15, 10:30] = 1  # object 0
    overlap[0, 20:35, 50:70] = 2  # object 1
    polys = augpreview._mask_polygons(overlap, 2)
    xs = np.array(polys[1][0::2])
    assert xs.min() > 0.6 and xs.max() < 0.9  # object 1 sits at x 50..70 of 80
    separate = np.zeros((2, 40, 80), np.uint8)
    separate[0, 5:15, 10:30] = 1
    polys = augpreview._mask_polygons(separate, 2)
    assert polys[0] is not None and polys[1] is None  # empty mask: nothing to draw


def test_segment_preview_draws_masks(yolo_seg, tmp_path):
    ds = formats.load(yolo_seg)
    data = augpreview.preview_data(ds, "segment", tmp_path)
    imgs = augpreview.samples(data, "segment", {"imgsz": 320}, n=2, seed=0)
    assert len(imgs) == 2 and imgs[0].size == (320, 320)
    imgs[0].save(tmp_path / "seg.png")
