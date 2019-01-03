import PIL.ImageOps
import PIL.ImageFilter
import PIL.ImageChops
import typing

from .panel_pipeline import PanelPipeline
from .process import make_binarized_image


class PagePipeline(object):
    def __init__(
            self,
            drawer,
            drawer_sr,
            image,
            reference_images,
            threshold_binary,
            threshold_line,
            panel_rects,
    ):
        self.drawer = drawer
        self.drawer_sr = drawer_sr
        self.image = image
        self.reference_images = reference_images
        self.panel_rects = panel_rects
        self.threshold_binary = threshold_binary
        self.threshold_line = threshold_line

        self._raw_image = image

    def process(self):
        panels = self._pre_process()
        drawn_panel_images = [panel.process() for panel in panels]
        return self._post_process(drawn_panel_images)

    def _pre_process(self)
        panels = []
        for reference_image, panel_rect in zip(self.reference_images, self.panel_rects):
            bw_panel = self._make_panel_image(self._raw_image, panel_rect)
            panel = PanelPipeline(
                drawer=self.drawer,
                drawer_sr=self.drawer_sr,
                image=bw_panel,
                reference_image=reference_image,
                threshold=self.threshold_binary,
            )
            panels.append(panel)

        return panels

    def _post_process(self, drawn_panel_images):
        raw_line = self._make_rawline_image(self._raw_image)

        bg = self._make_page_image(self._raw_image, drawn_panel_images, [[r[0], r[1]] for r in self.panel_rects])
        line = make_binarized_image(raw_line, self.threshold_line)

        output = self._make_overlayed_image(bg, line)
        return output

    @staticmethod
    def _make_rawline_image(bw_image, filter_size=5):
        bw = bw_image.convert('L')
        line_raw = bw.filter(PIL.ImageFilter.MaxFilter(filter_size))
        line_raw = PIL.ImageChops.difference(bw, line_raw)
        line_raw = PIL.ImageOps.invert(line_raw)
        return line_raw

    @staticmethod
    def _make_panel_image(base_image, panel_rect):
        width = panel_rect[2]
        height = panel_rect[3]
        img = base_image.crop((panel_rect[0], panel_rect[1], panel_rect[0] + width, panel_rect[1] + height))
        return img

    @staticmethod
    def _make_page_image(base_image, panel_image_list, offset_list):
        base = base_image.copy()
        for panel_image, offset in zip(panel_image_list, offset_list):
            base.paste(panel_image, tuple(offset))
        return base

    @staticmethod
    def _make_overlayed_image(page_image, line_image):
        img = page_image.copy()
        alpha = PIL.ImageOps.invert(line_image)
        img.paste(line_image, mask=alpha)
        return img
