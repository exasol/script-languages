import shutil
import tempfile
from pathlib import Path

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.image_info import ImageInfo
from exaslct_src.lib.docker.build_context_creator import BuildContextCreator
from exaslct_src.lib.docker.build_log_handler import BuildLogHandler
from exaslct_src.lib.docker.docker_image_creator_base_task import DockerImageCreatorBaseTask
from exaslct_src.lib.docker_config import docker_client_config, docker_build_arguments
from exaslct_src.lib.still_running_logger import StillRunningLogger


class DockerBuildImageTask(DockerImageCreatorBaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._low_level_client = docker_client_config().get_low_level_client()

    def __del__(self):
        self._low_level_client.close()

    def run_task(self):
        self.logger.info("Build docker image %s, log file can be found here %s",
                          self.image_info.get_target_complete_name(), self.get_log_path())
        temp_directory = tempfile.mkdtemp(prefix="script_langauge_container_tmp_dir",
                                          dir=build_config().temporary_base_directory)
        try:
            image_description = self.image_info.image_description
            build_context_creator = BuildContextCreator(temp_directory,
                                                        self.image_info,
                                                        self.get_log_path())
            build_context_creator.prepare_build_context_to_temp_dir()
            output_generator = \
                self._low_level_client.build(path=temp_directory,
                                             nocache=docker_client_config.no_cache,
                                             tag=self.image_info.get_target_complete_name(),
                                             rm=True,
                                             buildargs=dict(**image_description.transparent_build_arguments,
                                                            **image_description.image_changing_build_arguments,
                                                            **docker_build_arguments().secret))
            self._handle_output(output_generator, self.image_info)
        finally:
            shutil.rmtree(temp_directory)

    def _handle_output(self, output_generator, image_info: ImageInfo):
        log_file_path = Path(self.get_log_path(),"docker-build.log")
        with BuildLogHandler(log_file_path, self.logger, image_info) as log_hanlder:
            still_running_logger = StillRunningLogger(
                self.logger, "build image %s" % image_info.get_target_complete_name())
            for log_line in output_generator:
                still_running_logger.log()
                log_hanlder.handle_log_line(log_line)