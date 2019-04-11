from build_utils.lib.data.image_info import ImageInfo
from build_utils.lib.data.info import Info
from build_utils.release_type import ReleaseType


class ExportInfo(Info):

    def __init__(self, path: str, complete_name: str, name: str, hash: str, is_new: bool,
                 release_type:ReleaseType, depends_on_image: ImageInfo):
        self.release_type = release_type
        self.depends_on_image = depends_on_image
        self.is_new = is_new
        self.hash = hash
        self.name = name
        self.complete_name = complete_name
        self.path = path
