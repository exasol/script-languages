from enum import Enum
from typing import Dict, Any

from exaslct_src.lib.data.info import Info


class ImageState(Enum):
    NOT_EXISTING = 0,
    # After analyze phase or if build phase did touch the image
    NEEDS_TO_BE_BUILD = 1,
    TARGET_LOCALLY_AVAILABLE = 2,
    SOURCE_LOCALLY_AVAILABLE = 3,
    REMOTE_AVAILABLE = 4,
    CAN_BE_LOADED = 5,
    # After build phase
    WAS_BUILD = 6,
    USED_LOCAL = 7,
    WAS_PULLED = 8,
    WAS_LOADED = 9
    WAS_TAGED = 10

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

    def __init__(self,
                 source_repository_name: str, target_repository_name: str,
                 source_tag: str, target_tag: str, hash: str,
                 image_description: ImageDescription,
                 image_state: ImageState = ImageState.NOT_EXISTING,
                 depends_on_images: Dict[str, "ImageInfo"] = None):
        self.target_repository_name = target_repository_name
        self.source_repository_name = source_repository_name
        self.image_description = image_description
        if isinstance(image_state, ImageState):
            self.image_state = image_state.name
        elif isinstance(image_state,str):
            self.image_state = ImageState[image_state].name
        elif image_state is None:
            self.image_state = None
        else:
            raise TypeError(f"{type(image_state)} for image_state not supported")
        self.depends_on_images = depends_on_images
        self.source_tag = source_tag
        self.target_tag = target_tag
        self.hash = hash

    def get_target_complete_name(self):
        return f"{self.target_repository_name}:{self.get_target_complete_tag()}"

    def get_source_complete_name(self):
        return f"{self.source_repository_name}:{self.get_source_complete_tag()}"

    def get_source_complete_tag(self):
        if self.hash == "":
            return f"{self.source_tag}"
        else:
            return f"{self.source_tag}_{self.hash}"

    def get_target_complete_tag(self):
        if self.hash == "":
            return f"{self.target_tag}"
        else:
            return f"{self.target_tag}_{self.hash}"
