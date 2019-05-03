from enum import Enum
from typing import Dict, Any

from exaslct_src.lib.data.info import Info


class ImageState(Enum):
    NOT_EXISTING = 0,
    # After analyze phase or if build phase did touch the image
    NEEDS_TO_BE_BUILD = 1,
    LOCALLY_AVAILABLE = 2,
    REMOTE_AVAILABLE = 3,
    CAN_BE_LOADED = 4,
    # After build phase
    WAS_BUILD = 5,
    USED_LOCAL = 6,
    WAS_PULLED = 7,
    WAS_LOADED = 8


class ImageDescription:
    def __init__(self,
                 dockerfile: str,
                 image_changing_build_arguments: Dict[str, Any],
                 transparent_build_arguments: Dict[str, Any],
                 mapping_of_build_files_and_directories: Dict[str, str]):
        self.transparent_build_arguments = transparent_build_arguments
        self.image_changing_build_arguments = image_changing_build_arguments
        self.mapping_of_build_files_and_directories = mapping_of_build_files_and_directories
        self.dockerfile = dockerfile


class ImageInfo(Info):

    def __init__(self, complete_name: str, complete_tag:str,
                 name: str, tag: str, hash: str,
                 image_description: ImageDescription,
                 image_state: ImageState = ImageState.NOT_EXISTING,
                 depends_on_images: Dict[str, "ImageInfo"] = None):
        self.complete_tag = complete_tag
        self.image_description = image_description
        self.image_state = image_state.name
        self.depends_on_images = depends_on_images
        self.complete_name = complete_name
        self.name = name
        self.tag = tag
        self.hash = hash
