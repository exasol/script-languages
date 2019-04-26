from typing import List

from exaslct_src.lib.data.info import Info


class ImageInfo(Info):

    def __init__(self, complete_name: str, name: str, tag: str, hash: str = "",
                 was_build: bool = False, was_pulled: bool = False,
                 depends_on_images: List["ImageInfo"] = None):
        self.was_pulled = was_pulled
        self.was_build = was_build
        self.depends_on_images = depends_on_images
        self.complete_name = complete_name
        self.name = name
        self.tag = tag
        self.hash = hash
