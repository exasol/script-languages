import logging
from pathlib import Path
from typing import Dict

import docker
import git
import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector, \
    IMAGE_INFO
from exaslct_src.lib.data.image_info import ImageInfo, ImageState, ImageDescription
from exaslct_src.lib.docker.docker_image_target import DockerImageTarget
from exaslct_src.lib.docker.docker_registry_image_checker import DockerRegistryImageChecker
from exaslct_src.lib.docker_config import docker_client_config, docker_build_arguments
from exaslct_src.lib.stoppable_task import StoppableTask
from exaslct_src.lib.utils.build_context_hasher import BuildContextHasher


class DockerAnalyzeImageTask(StoppableTask):
    logger = logging.getLogger('luigi-interface')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source_repository_name = self.get_source_repository_name()
        self._target_repository_name = self.get_target_repository_name()
        self._source_image_tag = self.get_source_image_tag()
        self._target_image_tag = self.get_target_image_tag()
        merged_transparent_build_arguments = {**self.get_transparent_build_arguments(),
                                              **docker_build_arguments().transparent}
        merged_image_changing_build_arguments = {**self.get_image_changing_build_arguments(),
                                                 **docker_build_arguments().image_changing}
        self.image_description = ImageDescription(
            dockerfile=self.get_dockerfile(),
            mapping_of_build_files_and_directories=self.get_mapping_of_build_files_and_directories(),
            image_changing_build_arguments=merged_image_changing_build_arguments,
            transparent_build_arguments=merged_transparent_build_arguments
        )
        self._dockerfile = self.get_dockerfile()
        self._prepare_outputs()
        self._build_context_hasher = \
            BuildContextHasher(self.__repr__(),
                               self.image_description)
        self._client = docker_client_config().get_client()

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        self._image_info_target = luigi.LocalTarget(
            "%s/info/image/analyze/%s/%s"
            % (build_config().output_directory,
               self._target_repository_name, self._source_image_tag))
        if self._image_info_target.exists():
            self._image_info_target.remove()

    def get_source_repository_name(self) -> str:
        """
        Called by the constructor to get the image name for pulls. Sub classes need to implement this method.
        :return: image name
        """
        pass

    def get_target_repository_name(self) -> str:
        """
        Called by the constructor to get the image name for pushs. Sub classes need to implement this method.
        :return: image name
        """
        pass

    def get_source_image_tag(self) -> str:
        """
        Called by the constructor to get the image tag for pulls. Sub classes need to implement this method.
        :return: image tag
        """
        pass

    def get_target_image_tag(self) -> str:
        """
        Called by the constructor to get the image tag for pushs. Sub classes need to implement this method.
        :return: image tag
        """
        pass

    def get_mapping_of_build_files_and_directories(self) -> Dict[str, str]:
        """
        Called by the constructor to get the build files and directories mapping.
        The keys are the relative paths to the destination in build context and
        the values are the paths to the source directories or files.
        Sub classes need to implement this method.
        :return: dictionaries with destination path as keys and source paths in values
        """
        pass

    def get_dockerfile(self) -> str:
        """
        Called by the constructor to get the path to the dockerfile.
        Sub classes need to implement this method.
        :return: path to the dockerfile
        """
        pass

    def get_image_changing_build_arguments(self) -> Dict[str, str]:
        """
        Called by the constructor to get the path image changing docker build arguments.
        Different values for these arguments might change the image, such that they
        will be part of the image hash and can cause a build.
        A common use case is define a mirror from which packages should be installed.
        Sub classes need to implement this method.
        :return: Dictionary of build arguments, where the keys are the argument name
        """
        return dict()

    def get_transparent_build_arguments(self) -> Dict[str, str]:
        """
        Called by the constructor to get the path transparent docker build arguments.
        Transparent arguments do not change the contain of the images.
        They are not part of the image hash. A common use case is define a mirror
        from which packages should be installed.
        Sub classes need to implement this method.
        :return: Dictionary of build arguments, where the keys are the argument name
        """
        return dict()

    def is_rebuild_requested(self) -> bool:
        pass

    def output(self):
        return {IMAGE_INFO: self._image_info_target}

    def run_task(self):
        image_info_of_dependencies = DependencyImageInfoCollector().get_from_dict_of_inputs(self.input())
        image_hash = self._build_context_hasher.generate_image_hash(image_info_of_dependencies)
        image_info = ImageInfo(
            source_repository_name=self._source_repository_name,
            target_repository_name=self._target_repository_name,
            source_tag=self._source_image_tag,
            target_tag=self._target_image_tag,
            hash=image_hash,
            commit=self.get_commit_id(),
            build_name=build_config().build_name,
            depends_on_images=image_info_of_dependencies,
            image_state=None,
            image_description=self.image_description
        )
        target_image_target = DockerImageTarget(self._target_repository_name, image_info.get_target_complete_tag())
        source_image_target = DockerImageTarget(self._source_repository_name, image_info.get_source_complete_tag())
        image_state = self.get_image_state(source_image_target,
                                           target_image_target,
                                           image_info_of_dependencies)
        image_info.image_state = image_state.name  # TODO setter for image_state
        with self._image_info_target.open("w") as f:
            f.write(image_info.to_json())

    def get_commit_id(self):
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            return sha
        except Exception as e:
            self.logger.info("Task %s_ Not a Git Repository, can't determine the commit id for the image_info",
                             self.__repr__())
            return ""

    def get_image_state(self,
                        source_image_target: DockerImageTarget,
                        target_image_target: DockerImageTarget,
                        image_info_of_dependencies: Dict[str, ImageInfo]) -> ImageState:

        if self.is_rebuild_necessary(image_info_of_dependencies):
            return ImageState.NEEDS_TO_BE_BUILD
        else:
            if self.is_image_locally_available(target_image_target) \
                    and not build_config().force_pull \
                    and not build_config().force_load:
                return ImageState.TARGET_LOCALLY_AVAILABLE
            if self.is_image_locally_available(source_image_target) \
                    and not build_config().force_pull \
                    and not build_config().force_load:
                return ImageState.SOURCE_LOCALLY_AVAILABLE
            elif self.can_image_be_loaded(source_image_target):
                return ImageState.CAN_BE_LOADED
            elif self.is_image_in_registry(source_image_target):
                return ImageState.REMOTE_AVAILABLE
            else:
                return ImageState.NEEDS_TO_BE_BUILD

    def needs_any_dependency_to_be_build(
            self, image_info_of_dependencies: Dict[str, ImageInfo]) -> bool:
        return any(image_info.image_state == ImageState.NEEDS_TO_BE_BUILD
                   for image_info in image_info_of_dependencies.values())

    def is_rebuild_necessary(self, image_info_of_dependencies: Dict[str, ImageInfo]):
        return self.needs_any_dependency_to_be_build(image_info_of_dependencies) or \
               self.is_rebuild_requested()

    def is_image_locally_available(self, image_target: DockerImageTarget):
        exists=image_target.exists()
        self.logger.info(
                f"Task {self.__repr__()}: Checking if image {image_target.get_complete_name()} "
                f"is locally available, result {exists}")
        return exists

    def can_image_be_loaded(self, image_target: DockerImageTarget):
        if build_config().cache_directory is not None:
            image_path = self.get_path_to_cached_image(image_target)
            self.logger.info(f"Task {self.__repr__()}: Checking if image archive {image_path} "
                             f"is available in cache directory, "
                             f"result {image_path.exists()}")
            return image_path.exists()
        else:
            return False

    def get_path_to_cached_image(self, image_target):
        image_path = \
            Path(build_config().cache_directory) \
                .joinpath(Path(image_target.get_complete_name() + ".tar"))
        return image_path

    def is_image_in_registry(self, image_target: DockerImageTarget):
        try:
            self.logger.info("Task %s: Try to find image %s in registry", self.__repr__(),
                             image_target.get_complete_name())
            exists = DockerRegistryImageChecker().check(image_target.get_complete_name())
            if exists:
                self.logger.info("Task %s: Found image %s in registry", self.__repr__(),
                                 image_target.get_complete_name())
            return exists
        except docker.errors.DockerException as e:
            self.logger.warning("Task %s: Image %s not in registry, got exception %s", self.__repr__(),
                                image_target.get_complete_name(), e)
            return False
