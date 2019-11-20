from exaslct_src.lib.still_running_logger import StillRunningLogger
from exaslct_src.lib.docker.pull_log_handler import PullLogHandler
from exaslct_src.lib.docker.docker_image_creator_base_task import DockerImageCreatorBaseTask
from exaslct_src.lib.docker.docker_image_target import DockerImageTarget
from exaslct_src.lib.docker_config import source_docker_repository_config
import time

class DockerPullImageTask(DockerImageCreatorBaseTask):

    RETRIES = 3

    def run_task(self):
        image_target = DockerImageTarget(image_name=self.image_info.source_repository_name,
                                         image_tag=self.image_info.get_source_complete_tag())
        if source_docker_repository_config().username is not None and \
                source_docker_repository_config().password is not None:
            auth_config = {
                "username": source_docker_repository_config().username,
                "password": source_docker_repository_config().password
            }
        else:
            auth_config = None

        for i in range(self.RETRIES):
            try:
                self.pull(image_target,auth_config)
                break
            except Exception as e:
                if i<(self.RETRIES-1):
                    self.logger.exception("Error while pulling image: %s",e)
                    time.sleep(20)
                else:
                    raise e
                
        self.client.images.get(self.image_info.get_source_complete_name()).tag(
            repository=self.image_info.target_repository_name,
            tag=self.image_info.get_target_complete_tag()
        )

    def pull(self,image_target,auth_config):
        self.logger.info("Try to pull docker image %s", image_target.get_complete_name())
        output_generator = self.low_level_client.pull(
                repository=image_target.image_name, tag=image_target.image_tag,
                auth_config=auth_config,
                stream=True)
        self._handle_output(output_generator, self.image_info)


    def _handle_output(self, output_generator, image_info):
        log_file_path = self.get_log_path().joinpath("pull_docker_db_image.log")
        with PullLogHandler(log_file_path, self.logger, image_info) as log_hanlder:
            still_running_logger = StillRunningLogger(
                self.logger, "pull image %s" % image_info.get_source_complete_name())
            for log_line in output_generator:
                still_running_logger.log()
                log_hanlder.handle_log_line(log_line)

