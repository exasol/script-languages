import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime

import docker
import humanfriendly
import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.command_log_handler import CommandLogHandler
from exaslct_src.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from exaslct_src.lib.data.dependency_collector.dependency_release_info_collector import RELEASE_INFO
from exaslct_src.lib.data.image_info import ImageInfo
from exaslct_src.lib.data.release_info import ExportInfo
from exaslct_src.lib.docker_config import docker_config
from exaslct_src.lib.flavor import flavor
from exaslct_src.lib.still_running_logger import StillRunningLogger
from exaslct_src.lib.test_runner.create_export_directory import CreateExportDirectory
from exaslct_src.stoppable_task import StoppableTask
from exaslct_src.release_type import ReleaseType


# TODO create docker image from exported container and if possible fetch it from docker hub
#       required again the analysis of images without actual building them.
#       It is possible with docker import to get an images from the packed tar and with
#       docker export we get a tar with a single layer tar which contains the unchanged packed tar
class ExportContainerTask(StoppableTask):
    logger = logging.getLogger('luigi-interface')
    flavor_path = luigi.Parameter()
    export_path = luigi.OptionalParameter(None)
    release_name = luigi.OptionalParameter(None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._docker_config = docker_config()
        self._prepare_outputs()

    def _prepare_outputs(self):
        self.flavor_name = flavor.get_name_from_path(self.flavor_path)
        self.release_type_name = self.get_release_type().name
        self._target = luigi.LocalTarget(
            "%s/info/export/%s/%s"
            % (self._build_config.output_directory,
               self.flavor_name,
               self.release_type_name
               # datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
               ))
        if self._target.exists():
            self._target.remove()

    def output(self):
        return {RELEASE_INFO: self._target}

    def requires_tasks(self):
        return {"release_task": self.get_release_task(self.flavor_path),
                "export_directory": CreateExportDirectory()}

    def get_release_task(self, flavor_path):
        pass

    def get_release_type(self) -> ReleaseType:
        pass

    def run_task(self):
        image_infos = DependencyImageInfoCollector().get_from_dict_of_inputs(self.input())
        image_info_of_release_image = image_infos["release_task"]
        release_image_name = image_info_of_release_image.complete_name
        release_path = pathlib.Path(self.get_release_directory()).absolute()
        complete_name = f"""{image_info_of_release_image.tag}-{image_info_of_release_image.hash}"""
        cache_file = release_path.joinpath(complete_name + ".tar.gz").absolute()
        self.remove_release_file_if_requested(cache_file)

        is_new = False
        if not cache_file.exists():
            self.create_release(release_image_name, cache_file)
            is_new = True
        output_file = self.copy_cache_file_to_output_path(cache_file, is_new)
        self.write_release_info(image_info_of_release_image, is_new, cache_file, complete_name, output_file)

    def copy_cache_file_to_output_path(self, cache_file, is_new):
        output_file = None
        if self.export_path is not None:
            if self.release_name is not None:
                suffix = f"""_{self.release_name}"""
            else:
                suffix = ""
            file_name = f"""{self.flavor_name}_{self.release_type_name}{suffix}.tar.gz"""
            output_file = pathlib.Path(self.export_path).joinpath(file_name)
            if not output_file.exists() or is_new:
                shutil.copy2(cache_file, output_file)
        return output_file

    def get_release_directory(self):
        return pathlib.Path(self.input()["export_directory"].path).absolute().parent

    def remove_release_file_if_requested(self, release_file):
        if release_file.exists() and \
                (self._build_config.force_rebuild or
                 self._build_config.force_pull):
            self.logger.info("Task %s: Removed container file %s", self.task_id, release_file)
            os.remove(release_file)

    def write_release_info(self, image_info_of_release_image: ImageInfo, is_new: bool,
                           cache_file: pathlib.Path, release_name: str, output_file_path: pathlib.Path):
        if output_file_path is None:
            output_file = None
        else:
            output_file = str(output_file_path)
        release_info = ExportInfo(
            cache_file=str(cache_file),
            complete_name=release_name,
            name=self.flavor_name,
            hash=str(image_info_of_release_image.hash),
            is_new=is_new,
            depends_on_image=image_info_of_release_image,
            release_type=self.get_release_type(),
            release_name=str(self.release_name),
            output_file=output_file
        )
        json = release_info.to_json()
        with self.output()[RELEASE_INFO].open("w") as file:
            file.write(json)

    def create_release(self, release_image_name: str, release_file: str):
        self.logger.info("Task %s: Create container file %s", self.task_id, release_file)
        temp_directory = tempfile.mkdtemp(prefix="release_archive_",
                                          dir=self._build_config.temporary_base_directory)
        try:
            log_path = self.prepare_log_dir(release_image_name)
            export_file = self.create_and_export_container(release_image_name, temp_directory)
            extract_dir = self.extract_exported_container(log_path, export_file, temp_directory)
            self.modify_extracted_container(extract_dir)
            self.pack_release_file(log_path, extract_dir, release_file)
        finally:
            shutil.rmtree(temp_directory)

    def create_and_export_container(self, release_image_name: str, temp_directory: str):
        self.logger.info("Task %s: Export container %s", self.task_id, release_image_name)
        client = docker_config().get_client()
        try:
            container = client.containers.create(image=release_image_name)
            try:
                return self.export_container(container, release_image_name, temp_directory)
            finally:
                container.remove(force=True)
        finally:
            client.close()

    def export_container(self, container, release_image_name: str, temp_directory: str):
        generator = container.export(chunk_size=humanfriendly.parse_size("10mb"))
        export_file = temp_directory + "/export.tar"
        with open(export_file, "wb") as file:
            still_running_logger = StillRunningLogger(
                self.logger, self.task_id, "Export image %s" % release_image_name)
            for chunk in generator:
                still_running_logger.log()
                file.write(chunk)
        return export_file

    def pack_release_file(self, log_path: pathlib.Path, extract_dir: str, release_file: str):
        self.logger.info("Task %s: Pack container file %s", self.task_id, release_file)
        extract_content = " ".join("'%s'" % file for file in os.listdir(extract_dir))
        command = f"""tar -C '{extract_dir}' -cvzf '{release_file}' {extract_content}"""
        self.run_command(command, "packing container file %s" % release_file,
                         log_path.joinpath("pack_release_file.log"))

    def modify_extracted_container(self, extract_dir: str):
        os.symlink("/conf/resolv.conf", f"""{extract_dir}/etc/resolv.conf""")
        os.symlink("/conf/hosts", f"""{extract_dir}/etc/hosts""")

    def extract_exported_container(self, log_path: pathlib.Path, export_file: str, temp_directory: str):
        self.logger.info("Task %s: Extract exported file %s", self.task_id, export_file)
        extract_dir = temp_directory + "/extract"
        os.makedirs(extract_dir)
        excludes = " ".join(
            ["--exclude='%s'" % dir for dir in ["dev/*", "proc/*", "etc/resolv.conf", "etc/hosts"]])
        self.run_command(f"""tar {excludes} -xvf '{export_file}' -C '{extract_dir}'""",
                         "extracting exported container %s" % export_file,
                         log_path.joinpath("extract_release_file.log"))
        return extract_dir

    def prepare_log_dir(self, release_image_name: str):
        log_dir = pathlib.Path("%s/logs/exports/%s/%s/"
                               % (self._build_config.output_directory,
                                  release_image_name,
                                  datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
        if log_dir.exists():
            shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def run_command(self, command: str, description: str, log_file_path: pathlib.Path):
        with subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            with CommandLogHandler(log_file_path, self.logger, self.task_id, description) as log_handler:
                still_running_logger = StillRunningLogger(
                    self.logger, self.task_id, description)
                log_handler.handle_log_line((command + "\n").encode("utf-8"))
                for line in iter(process.stdout.readline, b''):
                    still_running_logger.log()
                    log_handler.handle_log_line(line)
                process.wait(timeout=60 * 2)
                return_code_log_line = "return code %s" % process.returncode
                log_handler.handle_log_line(return_code_log_line.encode("utf-8"), process.returncode != 0)
