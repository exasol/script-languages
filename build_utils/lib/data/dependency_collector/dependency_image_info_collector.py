from typing import Dict

from build_utils.lib.data.image_info import ImageInfo
from build_utils.lib.data.dependency_collector.dependency_collector import DependencyInfoCollector


class DependencyImageInfoCollector(DependencyInfoCollector[ImageInfo]):

    def is_info(self, input):
        return isinstance(input, Dict) and IMAGE_INFO in input

    def read_info(self, value) -> ImageInfo:
        with value[IMAGE_INFO].open("r") as file:
            return ImageInfo.from_json(file.read())


IMAGE_INFO = "image_info"
