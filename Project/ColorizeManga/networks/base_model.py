import chainer
import numpy


class BaseModel(chainer.Chain):
    def __call__(self, x, test=False):
        raise NotImplementedError

    def generate_rgb_image(self, gray_images, **kwargs):
        raise NotImplementedError
