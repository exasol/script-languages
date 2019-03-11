import pathlib
from typing import Dict

import luigi

from build_utils.docker_pull_or_build_image_tasks import DockerPullOrBuildImageTask


class DockerPullOrBuildFlavorImageTask(DockerPullOrBuildImageTask):
    flavor_path = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        self.build_step = self.get_build_step()
        self.additional_build_directories_mapping = self.get_additional_build_directories_mapping()
        super().__init__(*args, **kwargs)

    def get_build_step(self) -> str:
        pass

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        """
        Called by the constructor to get additional build directories which are specific to the build step.
        This mappings gets merged with the default flavor build directories mapping.
        The keys are the relative paths to the destination in build context and
        the values are the paths to the source directories or files.
        Sub classes need to implement this method.
        :return: dictionaries with destination path as keys and source paths in values
        """
        return {}

    def get_image_name(self) -> str:
        return "scripting-language-container"

    def get_image_tag(self) -> str:
        flavor_name = self.get_flavor_name()
        return "%s-%s" % (flavor_name, self.build_step)

    def get_flavor_name(self):
        path = pathlib.PurePath(self.flavor_path)
        flavor_name = path.name
        return flavor_name

    def get_build_directories_mapping(self) -> Dict[str, str]:
        result = {self.build_step: "%s/%s" % (self.flavor_path, self.build_step)}
        result.update(self.additional_build_directories_mapping)
        return result

    def get_dockerfile(self) -> str:
        return "resources/test-flavor/%s/Dockerfile" % (self.build_step)
