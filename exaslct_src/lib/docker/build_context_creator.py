import os
import pathlib
import shutil
from pathlib import Path

from jinja2 import Template

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.image_info import ImageInfo


class BuildContextCreator:
    def __init__(self,
                 build_config: build_config,
                 temp_directory,
                 image_info: ImageInfo,
                 log_file_path: Path):
        self._image_info = image_info
        self._log_file_path = log_file_path
        self._image_info_of_dependencies = self._image_info.depends_on_images
        self._image_description = self._image_info.image_description
        self._temp_directory = temp_directory

    def prepare_build_context_to_temp_dir(self):
        self._copy_build_files_and_directories()
        self._prepare_dockerfile()
        self._log_build_context()

    def _prepare_dockerfile(self):
        with open(self._image_description.dockerfile, "rt") as file:
            dockerfile_content = file.read()
        template = Template(dockerfile_content)
        image_names_of_dependencies = \
            {key: image_info.complete_name
             for key, image_info
             in self._image_info_of_dependencies.items()}
        final_dockerfile = template.render(**image_names_of_dependencies)
        with open(self._temp_directory + "/Dockerfile", "wt") as file:
            file.write(final_dockerfile)

    def _copy_build_files_and_directories(self):
        for dest, src in self._image_description.mapping_of_build_files_and_directories.items():
            src_path = pathlib.Path(src)
            dest_path = self._temp_directory + "/" + dest
            if src_path.is_dir():
                shutil.copytree(src, dest_path)
            elif src_path.is_file():
                shutil.copy2(src, dest_path)
            else:
                raise Exception("Source path %s is neither a file or a directory" % src)

    def _log_build_context(self):
        if build_config().log_build_context_content:
            build_context_log_file = self._log_file_path.joinpath("docker-build-context.log")
            with build_context_log_file.open("wt") as log_file:
                for file in self._get_files_in_build_context(self._temp_directory):
                    log_file.write(file)
                    log_file.write("\n")
            dockerfile_log_file = self._log_file_path.joinpath("Dockerfile.generated")
            shutil.copy(self._temp_directory + "/Dockerfile", str(dockerfile_log_file))

    def _get_files_in_build_context(self, temp_directory):
        return [os.path.join(r, file) for r, d, f in os.walk(temp_directory) for file in f]