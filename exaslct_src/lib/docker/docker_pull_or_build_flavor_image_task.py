from pathlib import Path
from typing import Dict

import luigi

from exaslct_src.lib.docker.docker_pull_or_build_image_tasks import DockerPullOrBuildImageTask
from exaslct_src.lib.docker_config import docker_config
from exaslct_src.lib.flavor import flavor


class DockerPullOrBuildFlavorImageTask(DockerPullOrBuildImageTask):
    flavor_path = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        self.build_step = self.get_build_step()
        self.additional_build_directories_mapping = self.get_additional_build_directories_mapping()
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()

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

    def get_path_in_flavor(self):
        return None

    def get_image_name(self) -> str:
        return self._docker_config.repository_name

    def get_image_tag(self) -> str:
        flavor_name = flavor.get_name_from_path(self.flavor_path)
        return "%s-%s" % (flavor_name, self.build_step)

    def get_mapping_of_build_files_and_directories(self) -> Dict[str, str]:
        build_step_path = self.get_build_step_path()
        result = {self.build_step: str(build_step_path)}
        result.update(self.additional_build_directories_mapping)
        return result

    def get_build_step_path(self):
        path_in_flavor = self.get_path_in_flavor()
        if path_in_flavor is None:
            build_step_path_in_flavor = Path(self.build_step)
        else:
            build_step_path_in_flavor = Path(path_in_flavor).joinpath(self.build_step)
        build_step_path = Path(self.flavor_path).joinpath(build_step_path_in_flavor)
        return build_step_path

    def get_dockerfile(self) -> str:
        return str(self.get_build_step_path().joinpath("Dockerfile"))
