from typing import Sequence
import numpy as np
from numpy.core.numeric import normalize_axis_tuple


def normalize_axis_list(axis, ndim):
    return list(normalize_axis_tuple(axis, ndim))


def get_random_box(image_size: Sequence[int], box_size: Sequence[int]) -> np.ndarray:
    image_size = np.array(image_size)
    box_size = np.array(box_size)

    assert np.all(image_size >= box_size)

    min_start = 0
    max_start = image_size - box_size
    start = np.random.randint(min_start, max_start + 1)
    box = np.array([start, start + box_size])

    return box
