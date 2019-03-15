from typing import Dict, List

from luigi import LocalTarget

from build_utils.image_info import ImageInfo


class ImageDependencyCollector:
    def get_dict_of_image_info_of_dependencies(self, input:Dict[str, Dict[str, LocalTarget]]) -> Dict[str, ImageInfo]:
        """
        Reads from input the image names produced by dependent tasks.
        :return Dictionary with dependency keys defined by requires method and images names as values
        """
        if isinstance(input, Dict):
            return {key: self.read_image_info(value)
                    for key, value in input.items()
                    if isinstance(value, Dict) and "image_info" in value}
        else:
            return dict()

    def get_list_of_image_info_of_dependencies(self, input:List[Dict[str, LocalTarget]]) -> List[ImageInfo]:
        """
        Reads from input the image names produced by dependent tasks.
        :return List of images names
        """
        if isinstance(input, Dict):
            return [self.read_image_info(value)
                    for value in input
                    if isinstance(value, Dict) and "image_info" in value]
        else:
            return list()

    def get_image_info_of_dependency(self, input:Dict[str, LocalTarget]) -> ImageInfo:
        """
        Reads from input the image names produced by dependent tasks.
        :return List of images names
        """
        if isinstance(input, Dict):
            if isinstance(input, Dict) and "image_info" in input:
                return self.read_image_info(input)
        return None


    def read_image_info(self, value):
        with value["image_info"].open("r") as file:
            return ImageInfo.from_json(file.read())