from typing import List

from exaslct_src.lib.data.info import Info


class ImageInfo(Info):

    def __init__(self, complete_name: str, name: str, tag: str, hash: str="",
                 is_new: bool=False, depends_on_images: List["ImageInfo"]=None):
        self.depends_on_images = depends_on_images
        self.is_new = is_new #TODO distinct is_new into was_pulled and was_build
        self.complete_name = complete_name
        self.name = name
        self.tag = tag
        self.hash = hash

