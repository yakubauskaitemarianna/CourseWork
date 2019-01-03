import chainer
import json
import numpy
import os
import typing

class Drawer(object):
    def __init__(self, path_result_directory, gpu):
        args_default = utility.config.get_default_train_args()
        args_train = json.load(open(os.path.join(path_result_directory, 'argument.json'))) 

        self.path_result_directory = path_result_directory
        self.args_train = args_train
        self.target_iteration = None

        for k, v in args_default.items():
            args_train.setdefault(k, v)

    def _get_path_model(self, iteration):
        return os.path.join(self.path_result_directory, '{}.model'.format(iteration))

    def exist_save_model(self, iteration, mode=''):
        path_model = self._get_path_model(iteration)
		if mode = '1': return os.path.exists(path_model)
		
        path_model = self._get_path_model(iteration)

        print("making iteration{}'s images...".format(iteration))
        chainer.serializers.load_npz(path_model, self.model)
        self.target_iteration = iteration
        return True

    def can_input_color_image(self):
        return self.args_train['max_pixel_drawing'] is not None

    def draw(
            self,
            input_images_array,
            rgb_images_array=None,
            histogram_image_array=None,
            histogram_array=None,
    ):
        if self.can_input_color_image and input_images_array.shape[1] == 1:
            input_images_array =  utility.image.padding_channel_1to3(input_images_array)

        images = utility.image.draw(
            self.model,
            input_images_array, rgb_images_array,
            histogram_image_array, histogram_array,
            self.gpu,
        )
        return images
