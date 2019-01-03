import numpy
from skimage.color import rgb2lab

def normalize(array, in_min, in_max, out_min, out_max):
    in_range = in_max - in_min
    out_range = out_max - out_min
    return (array - in_min) / in_range * out_range + out_min


def normalize_each_channel(array, in_min, in_max, out_min, out_max, split, concat):
    channels = tuple(
        normalize(channel, in_min[i], in_max[i], out_min[i], out_max[i])
        for i, channel in enumerate(split(array))
    )
    return concat(channels)


def normalize_zero_one(array, in_type, split, concat):
    out_min = (0, 0, 0)
    out_max = (1, 1, 1)

    if in_type == 'RGB':
        in_min = (0, 0, 0)
        in_max = (255, 255, 255)
    elif in_type == 'Lab':
        in_min, in_max = lab_min_max
    elif in_type == 'ab':
        in_min, in_max = lab_min_max
        in_min = in_min[1:]
        in_max = in_max[1:]
        out_min = out_min[1:]
        out_max = out_max[1:]
    else:
        raise ValueError(in_type)

    return normalize_each_channel(array, in_min, in_max, out_min, out_max, split, concat)
