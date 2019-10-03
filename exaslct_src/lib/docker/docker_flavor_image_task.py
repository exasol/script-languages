from pathlib import Path
from typing import Dict

import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.docker.docker_analyze_task import DockerAnalyzeImageTask
from exaslct_src.lib.docker_config import source_docker_repository_config, target_docker_repository_config
from exaslct_src.lib.flavor import flavor

# TODO change task inheritance with composition.
#  In this case DockerPullOrBuildFlavorImageTask could create a DockerPullOrBuildImageTask
#  if this would have parameters instead of abstract methods
from exaslct_src.lib.flavor_task import FlavorBaseTask


class DockerFlavorAnalyzeImageTask(DockerAnalyzeImageTask, FlavorBaseTask):

    def __init__(self, *args, **kwargs):
        self.build_step = self.get_build_step()
        self.additional_build_directories_mapping = self.get_additional_build_directories_mapping()
        super().__init__(*args, **kwargs)

    def is_rebuild_requested(self) -> bool:
        config = build_config()
        return (
                config.force_rebuild and
                (
                        self.get_build_step() in config.force_rebuild_from or
                        len(config.force_rebuild_from) == 0
                ))

    def get_build_step(self) -> str:
        """
        Called by the constructor to get the name of build step.
        Sub classes need to implement this method.
        :return: dictionaries with destination path as keys and source paths in values
        """
        pass

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        """
        Called by the constructor to get additional build directories or files which are specific to the build step.
        This mappings gets merged with the default flavor build directories mapping.
        The keys are the relative paths to the destination in build context and
        the values are the paths to the source directories or files.
        Sub classes need to implement this method.
        :return: dictionaries with destination path as keys and source paths in values
        """
        return {}

    def get_path_in_flavor(self):
        """
        Called by the constructor to get the path to the build context of the build step within the flavor path.
        Sub classes need to implement this method.
        :return: dictionaries with destination path as keys and source paths in values
        """
        return None

    def get_source_repository_name(self) -> str:
        return source_docker_repository_config().repository_name

    def get_target_repository_name(self) -> str:
        return target_docker_repository_config().repository_name

    def get_source_image_tag(self):
        if source_docker_repository_config().tag_prefix != "":
            return f"{source_docker_repository_config().tag_prefix}_{self.get_image_tag()}"
        else:
            return f"{self.get_image_tag()}"

    def get_target_image_tag(self):
        if target_docker_repository_config().tag_prefix != "":
            return f"{target_docker_repository_config().tag_prefix}_{self.get_image_tag()}"
        else:
            return f"{self.get_image_tag()}"

    def get_image_tag(self) -> str:
        flavor_name = self.get_flavor_name()
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
