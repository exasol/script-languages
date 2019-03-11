import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict

from docker import APIClient
from jinja2 import Template

from build_utils.docker_image_target import DockerImageTarget


class DockerImageBuilder:
    logger = logging.getLogger('luigi-interface')

    def __init__(self, task_id: str,
                 docker_base_url,
                 build_context_base_directory,
                 build_directories_mapping,
                 dockerfile,
                 log_build_context_content,
                 log_file_path: Path):
        self._low_level_client = APIClient(base_url=docker_base_url)
        self.task_id = task_id
        self.build_context_base_directory = build_context_base_directory
        self.build_directories_mapping = build_directories_mapping
        self.dockerfile = dockerfile
        self.log_build_context_content = log_build_context_content
        self.log_file_path = log_file_path

    def __del__(self):
        self._low_level_client.close()

    def build(self, image_target: DockerImageTarget, images_names_of_dependencies: Dict[str, str]):
        self.logger.info("Task %s: Execute build of image %s, config file can be found here %s ",
                         self.task_id, image_target.get_complete_name(), self.log_file_path)
        try:
            temp_directory = tempfile.mkdtemp(prefix="script_langauge_container_tmp_dir",
                                              dir=self.build_context_base_directory)
            self._prepare_build_context_to_temp_dir(temp_directory, images_names_of_dependencies)

            output_generator = \
                self._low_level_client.build(path=temp_directory,
                                             tag=image_target.get_complete_name(),
                                             rm=True)
            self._write_output_log(output_generator)
        finally:
            shutil.rmtree(temp_directory)

    def _write_output_log(self, output_generator):
        path = self.log_file_path.joinpath("docker-build.log")
        with path.open("wb") as log_file:
            if output_generator is not None:
                for log_line in output_generator:
                    log_file.write(log_line)

    def _prepare_build_context_to_temp_dir(self, temp_directory, images_names_of_dependencies: Dict[str, str]):
        self._copy_build_directories(temp_directory)
        self._prepare_dockerfile(temp_directory, images_names_of_dependencies)
        self._log_build_context(temp_directory)

    def _prepare_dockerfile(self, temp_directory, images_names_of_dependencies: Dict[str, str]):
        with open(self.dockerfile, "rt") as file:
            dockerfile_content = file.read()
        template = Template(dockerfile_content)
        final_dockerfile = template.render(**images_names_of_dependencies)
        with open(temp_directory + "/Dockerfile", "wt") as file:
            file.write(final_dockerfile)

    def _copy_build_directories(self, temp_directory):
        for dest, src in self.build_directories_mapping.items():
            shutil.copytree(src, temp_directory + "/" + dest)

    def _log_build_context(self, temp_directory):
        if self.log_build_context_content:
            build_context_log_file = self.log_file_path.joinpath("docker-build-context.log")
            with build_context_log_file.open("wt") as log_file:
                for file in self._get_files_in_build_context(temp_directory):
                    log_file.write(file)
                    log_file.write("\n")
            dockerfile_log_file = self.log_file_path.joinpath("Dockerfile.generated")
            shutil.copy(temp_directory + "/Dockerfile", str(dockerfile_log_file))

    def _get_files_in_build_context(self, temp_directory):
        return [os.path.join(r, file) for r, d, f in os.walk(temp_directory) for file in f]
