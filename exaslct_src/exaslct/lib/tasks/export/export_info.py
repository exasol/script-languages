from exasol_integration_test_docker_environment.lib.base.info import Info
from exasol_integration_test_docker_environment.lib.docker.images.image_info import ImageInfo


class ExportInfo(Info):

    def __init__(self, cache_file: str, complete_name: str, name: str, _hash: str, is_new: bool,
                 release_goal: str, depends_on_image: ImageInfo,
                 output_file: str = None, release_name: str = None):
        self.release_name = release_name
        self.output_file = output_file
        self.release_goal = release_goal
        self.depends_on_image = depends_on_image
        self.is_new = is_new
        self.hash = _hash
        self.name = name
        self.complete_name = complete_name
        self.cache_file = cache_file
