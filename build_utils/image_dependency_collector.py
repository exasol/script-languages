from typing import Dict

from luigi import LocalTarget

from build_utils.image_info import ImageInfo


class ImageDependencyCollector:
    def get_image_info_of_dependencies(self, input:Dict[str,Dict[str,LocalTarget]]) -> Dict[str, ImageInfo]:
        """
        Reads from input the image names produced by dependent tasks.
        :return Dictionary with dependency keys defined by requires method and images names as values
        """
        if isinstance(input, Dict):
            return {key: self.get_image_info_of_depedency(value)
                    for key, value in input.items()
                    if isinstance(value, Dict) and "image_info" in value}
        else:
            return dict()

    def get_image_info_of_depedency(self, value):
        with value["image_info"].open("r") as file:
            return ImageInfo.from_json(file.read())