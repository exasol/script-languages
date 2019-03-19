from typing import List

from build_utils.lib.data.info import Info


class ImageInfo(Info):

    def __init__(self, complete_name: str, name: str, tag: str, hash: str,
                 is_new: bool, depends_on_images: List["ImageInfo"]):
        self.depends_on_images = depends_on_images
        self.is_new = is_new
        self.complete_name = complete_name
        self.name = name
        self.tag = tag
        self.hash = hash

