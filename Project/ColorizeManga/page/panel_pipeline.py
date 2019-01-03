import numpy
from PIL import Image
from skimage.color import rgb2lab

from .process import make_binarized_image


def _calc_input_panel_rect(panel_size, input_width):
    w, h = panel_size
    scale = min(input_width / w, input_width / h)

    w, h = (round(w * scale), round(h * scale))
    x, y = (input_width - w) // 2, (input_width - h) // 2
    return [x, y, x + w, y + h]


def _make_input_panel_image(panel_image, input_panel_rect, input_width):
    x, y, _w, _h = (int(value) for value in input_panel_rect)
    w, h = _w - x, _h - y

    img = panel_image.convert('L')
    img = img.resize((w, h), Image.BICUBIC)

    bg = Image.new('RGB', (input_width, input_width), '#ffffff')
    bg.paste(img, (x, y))
    return bg


class PanelPipeline(object):
    def __init__(
            self,
            drawer,
            drawer_sr,
            image,
            reference_image,
            resize_width=224,
            threshold=200,
    ):
        self.drawer = drawer
        self.drawer_sr = drawer_sr
        self.image = image
        self.reference_image = reference_image
        self.resize_width = resize_width
        self.threshold = threshold

        self._crop_pre = None  

    def process(self):
        small_input_image, big_input_image = self._pre_process()
        drawn_panel_image = self._draw_process(small_input_image, big_input_image)
        return self._post_process(drawn_panel_image)

    def _pre_process(self):
        small_crop_pre = _calc_input_panel_rect(
            panel_size=self.image.size,
            input_width=self.resize_width,
        )

        input_panel_image = _make_input_panel_image(
            panel_image=self.image,
            input_panel_rect=small_crop_pre,
            input_width=self.resize_width,
        )

        if self.drawer_sr is not None:
            self._crop_pre = _calc_input_panel_rect(
                panel_size=self.image.size,
                input_width=self.resize_width * 2,
            )
        else:
            self._crop_pre = _calc_input_panel_rect(
                panel_size=self.image.size,
                input_width=self.resize_width,
            )

        small_input_image = make_binarized_image(input_panel_image, self.threshold)
        big_input_image = _make_input_panel_image(self.image, self._crop_pre, self.resize_width * 2)

        return small_input_image, big_input_image

    def _draw_process(self, small_input_image, big_input_image):
        lab = rgb2lab(numpy.array(small_input_image))
        lab[:, :, 0] /= 100
        small_image = self.drawer.draw(
            input_images_array=lab.astype(numpy.float32).transpose(2, 0, 1)[numpy.newaxis],
            rgb_images_array=numpy.array(self.reference_image, dtype=numpy.float32).transpose(2, 0, 1)[numpy.newaxis],
        )[0]

        small_image = small_image.convert('RGB')

        if self.drawer_sr is not None:
            drawn_panel_image = self._superresolution_process(small_image, big_input_image)
        else:
            drawn_panel_image = small_image

        return drawn_panel_image

    def _superresolution_process(self, small_image, big_input_image):
        small_array = numpy.array(small_image, dtype=numpy.float64)
        small_array = rgb2lab(small_array / 255).astype(numpy.float32)
        small_array = small_array.transpose(2, 0, 1) / 100
        concat_image_process = self.drawer_sr.colorization.get_concat_process()
        large_array = concat_image_process(big_input_image, test=True)
        sr_drawn_panel_image = self.drawer_sr.draw_only_super_pixel(
            image=small_array[numpy.newaxis],
            concat=large_array[numpy.newaxis],
        )[0]

        return sr_drawn_panel_image

    def _post_process(self, drawn_panel_image):
        array = numpy.array(drawn_panel_image)
        th = 255 / 6 / 2
        array[(array < th).all(axis=2)] = numpy.ones(3) * 0
        array[(array > 255 - th).all(axis=2)] = numpy.ones(3) * 255
        image = Image.fromarray(array)
        image = image.crop(self._crop_pre)
        image = image.resize(self.image.size, Image.BICUBIC)

        return image
