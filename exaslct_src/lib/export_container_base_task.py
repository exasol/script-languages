import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import humanfriendly
import luigi

from exaslct_src.lib.base.base_task import BaseTask
from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask
from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.command_log_handler import CommandLogHandler
from exaslct_src.lib.data.image_info import ImageInfo
from exaslct_src.lib.data.release_info import ExportInfo
from exaslct_src.lib.docker_config import docker_client_config
from exaslct_src.lib.flavor import flavor
from exaslct_src.lib.flavor_task import FlavorBaseTask
from exaslct_src.lib.still_running_logger import StillRunningLogger
from exaslct_src.lib.test_runner.create_export_directory import CreateExportDirectory


class ExportContainerBaseTask(FlavorBaseTask):
    logger = logging.getLogger('luigi-interface')
    export_path = luigi.OptionalParameter(None)
    release_name = luigi.OptionalParameter(None)
    release_goal = luigi.Parameter(None)

    def register_required(self):
        self._export_directory_future = self.register_dependency(CreateExportDirectory())
        self._release_task_future = self.register_dependency(self.get_release_task())

    def get_release_task(self) -> BaseTask:
        pass

    def run_task(self):
        image_info_of_release_image = self._release_task_future.get_output()  # type: ImageInfo
        cache_file, release_complete_name, release_image_name = \
            self._get_cache_file_path(image_info_of_release_image)
        self._remove_cached_exported_file_if_requested(cache_file)

        is_new = False
        if not cache_file.exists():
            self._export_release(release_image_name, cache_file)
            is_new = True
        output_file = self._copy_cache_file_to_output_path(cache_file, is_new)
        export_info = self._create_export_info(image_info_of_release_image,
                                               release_complete_name,
                                               cache_file, is_new, output_file)
        self.return_object(export_info)

    def _create_export_info(self, image_info_of_release_image: ImageInfo,
                            release_complete_name: str,
                            cache_file: Path, is_new: bool, output_file: Path):
        export_info = ExportInfo(
            cache_file=str(cache_file),
            complete_name=release_complete_name,
            name=self.get_flavor_name(),
            hash=str(image_info_of_release_image.hash),
            is_new=is_new,
            depends_on_image=image_info_of_release_image,
            release_goal=str(self.release_goal),
            release_name=str(self.release_name),
            output_file=str(output_file)
        )
        return export_info

    def _get_cache_file_path(self, image_info_of_release_image):
        release_image_name = image_info_of_release_image.get_target_complete_name()
        export_path = Path(self._export_directory_future.get_output()).absolute()
        release_complete_name = f"""{image_info_of_release_image.target_tag}-{image_info_of_release_image.hash}"""
        cache_file = Path(export_path, release_complete_name + ".tar.gz").absolute()
        return cache_file, release_complete_name, release_image_name

    def _copy_cache_file_to_output_path(self, cache_file, is_new):
        output_file = None
        if self.export_path is not None:
            if self.release_name is not None:
                suffix = f"""_{self.release_name}"""
            else:
                suffix = ""
            file_name = f"""{self.get_flavor_name()}_{self.release_goal}{suffix}.tar.gz"""
            output_file = Path(self.export_path, file_name)
            if not output_file.exists() or is_new:
                output_file.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(cache_file, output_file)
        return output_file

    def _remove_cached_exported_file_if_requested(self, release_file):
        if release_file.exists() and \
                (build_config().force_rebuild or
                 build_config().force_pull):
            self.logger.info("Removed container file %s", release_file)
            os.remove(release_file)

    def _export_release(self, release_image_name: str, release_file: str):
        self.logger.info("Create container file %s", release_file)
        temp_directory = tempfile.mkdtemp(prefix="release_archive_",
                                          dir=build_config().temporary_base_directory)
        try:
            log_path = self.get_log_path()
            export_file = self._create_and_export_container(release_image_name, temp_directory)
            extract_dir = self._extract_exported_container(log_path, export_file, temp_directory)
            self._modify_extracted_container(extract_dir)
            self._pack_release_file(log_path, extract_dir, release_file)
        finally:
            shutil.rmtree(temp_directory)

    def _create_and_export_container(self, release_image_name: str, temp_directory: str):
        self.logger.info("Export container %s", release_image_name)
        client = docker_client_config().get_client()
        try:
            container = client.containers.create(image=release_image_name)
            try:
                return self._export_container(container, release_image_name, temp_directory)
            finally:
                container.remove(force=True)
        finally:
            client.close()

    def _export_container(self, container, release_image_name: str, temp_directory: str):
        generator = container.export(chunk_size=humanfriendly.parse_size("10mb"))
        export_file = temp_directory + "/export.tar"
        with open(export_file, "wb") as file:
            still_running_logger = StillRunningLogger(
                self.logger, "Export image %s" % release_image_name)
            for chunk in generator:
                still_running_logger.log()
                file.write(chunk)
        return export_file

    def _pack_release_file(self, log_path: Path, extract_dir: str, release_file: str):
        self.logger.info("Pack container file %s", release_file)
        extract_content = " ".join("'%s'" % file for file in os.listdir(extract_dir))
        command = f"""tar -C '{extract_dir}' -cvzf '{release_file}' {extract_content}"""
        self.run_command(command, "packing container file %s" % release_file,
                         log_path.joinpath("pack_release_file.log"))

    def _modify_extracted_container(self, extract_dir: str):
        os.symlink("/conf/resolv.conf", f"""{extract_dir}/etc/resolv.conf""")
        os.symlink("/conf/hosts", f"""{extract_dir}/etc/hosts""")

    def _extract_exported_container(self, log_path: Path, export_file: str, temp_directory: str):
        self.logger.info("Extract exported file %s", export_file)
        extract_dir = temp_directory + "/extract"
        os.makedirs(extract_dir)
        excludes = " ".join(
            ["--exclude='%s'" % dir for dir in ["dev/*", "proc/*", "etc/resolv.conf", "etc/hosts"]])
        self.run_command(f"""tar {excludes} -xvf '{export_file}' -C '{extract_dir}'""",
                         "extracting exported container %s" % export_file,
                         log_path.joinpath("extract_release_file.log"))
        return extract_dir

    def run_command(self, command: str, description: str, log_file_path: Path):
        with subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            with CommandLogHandler(log_file_path, self.logger, description) as log_handler:
                still_running_logger = StillRunningLogger(
                    self.logger, description)
                log_handler.handle_log_line((command + "\n").encode("utf-8"))
                for line in iter(process.stdout.readline, b''):
                    still_running_logger.log()
                    log_handler.handle_log_line(line)
                process.wait(timeout=60 * 2)
                return_code_log_line = "return code %s" % process.returncode
                log_handler.handle_log_line(return_code_log_line.encode("utf-8"), process.returncode != 0)
