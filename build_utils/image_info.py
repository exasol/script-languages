import json
from typing import Dict, Set, List


class ImageInfo:

    @classmethod
    def create(cls, complete_name: str, name: str, tag: str, hash: str, is_new: bool,
               dependencies: Dict[str, "ImageInfo"]):
        depends_on_images = cls.merge_dependencies(dependencies)
        return cls(complete_name=complete_name, name=name, tag=tag, hash=hash,
                   is_new=is_new, depends_on_images=depends_on_images)

    @classmethod
    def merge_dependencies(cls, dependencies: Dict[str, "ImageInfo"]):
        depends_on_images = \
            list(set(
                [image
                 for dependency in dependencies.values()
                 for image in dependency.depends_on_images] +
                [dependency.complete_name
                 for dependency in dependencies.values()]))
        return depends_on_images

    def __init__(self, **kwargs):
        self.depends_on_images = kwargs["depends_on_images"]
        self.is_new = kwargs["is_new"]
        self.complete_name = kwargs["complete_name"]
        self.name = kwargs["name"]
        self.tag = kwargs["tag"]
        self.hash = kwargs["hash"]

    def to_json(self, indent=1):
        return json.dumps(self.__dict__, indent=indent)

    @classmethod
    def from_json(self, json_string):
        load = json.loads(json_string)
        return ImageInfo(**load)
