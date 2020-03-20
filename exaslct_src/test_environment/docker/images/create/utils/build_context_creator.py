import os
import pathlib
import shutil
import textwrap
from pathlib import Path

from jinja2 import Template

from exaslct_src.exaslct.lib.config.build_config import build_config
from exaslct_src.test_environment.docker.images.image_info import ImageInfo, ImageState


class BuildContextCreator:
    def __init__(self,
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
        self._prepare_image_info()
        self._log_build_context()

    def _prepare_image_info(self):
        self._image_info.image_state = ImageState.WAS_BUILD.name
        with open(self._temp_directory + "/image_info", "wt") as file:
            file.write(self._image_info.to_json())

    def _prepare_dockerfile(self):
        with open(self._image_description.dockerfile, "rt") as file:
            dockerfile_content = file.read()
        template = Template(dockerfile_content)
        image_names_of_dependencies = \
            {key: image_info.get_target_complete_name()
             for key, image_info
             in self._image_info_of_dependencies.items()}
        final_dockerfile = template.render(**image_names_of_dependencies)
        final_dockerfile += textwrap.dedent(f"""
        RUN mkdir -p /build_info/image_info
        COPY image_info /build_info/image_info/{self._image_info.target_tag}
        RUN mkdir -p /build_info/dockerfiles
        COPY Dockerfile /build_info/dockerfiles/{self._image_info.target_tag}
        """)
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
