from ..models.base_model import BaseModel
from ..utility.image import *


class BaseSaveImageExtension(chainer.training.Extension):
    @staticmethod
    def _save_images_with_trainer(
            trainer,
            images,
            prefix_directory,
            prefix_filename,
    ):
        path_directory = os.path.join(trainer.out, prefix_directory.format(trainer))
        prefix_filename = prefix_filename.format(trainer)
        return save_images(images=images, path_directory=path_directory, prefix_filename=prefix_filename)


class SaveGeneratedImageExtension(BaseSaveImageExtension):
    def __init__(
            self,
            gray_images_array,
            rgb_images_array,
            model,
            prefix_directory,
            prefix_filename="",
            image_mode='RGB',
    ):
        self.model = model
        self.prefix_directory = prefix_directory
        self.prefix_filename = prefix_filename
        self.image_mode = image_mode

        if self.model._cpu:
            self.gray_images_array = chainer.cuda.to_cpu(gray_images_array)
            self.rgb_images_array = chainer.cuda.to_cpu(rgb_images_array)

    def __call__(self, trainer):
        images = self.model.generate_rgb_image(self.gray_images_array, rgb_images_array=self.rgb_images_array)
        self._save_images_with_trainer(trainer, images, self.prefix_directory, self.prefix_filename)


class SaveRawImageExtension(BaseSaveImageExtension):
    def __init__(
            self,
            images_array,
            prefix_directory,
            prefix_filename="",
            image_mode='RGB',
            linedrawing=None,
    ):
        self.prefix_directory = prefix_directory
        self.prefix_filename = prefix_filename
        self.images_array = chainer.cuda.to_cpu(images_array)
        self.image_mode = image_mode
        self.linedrawing = linedrawing

    def __call__(self, trainer):
        if self.image_mode == 'gray':
            images = array_to_image(
                gray_images_array=self.images_array,
                mode=self.image_mode,
                linedrawing=self.linedrawing,
            )
        else:
            images = array_to_image(
                color_images_array=self.images_array,
                mode=self.image_mode,
                linedrawing=self.linedrawing,
            )
        self._save_images_with_trainer(trainer, images, self.prefix_directory, self.prefix_filename)