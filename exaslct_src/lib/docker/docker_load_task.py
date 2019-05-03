from pathlib import Path

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.docker.docker_image_creator_base_task import DockerImageCreatorBaseTask


class DockerLoadImageTask(DockerImageCreatorBaseTask):

    def run_task(self):
        image_archive_path = Path(build_config().cache_directory) \
            .joinpath(self.image_info.complete_name + ".tar")
        self.logger.info("Task %s: Try to load docker image %s from %s",
                         self.task_id, self.image_info.complete_name,image_archive_path)
        with image_archive_path.open("rb") as f:
            self.client.images.load(f)
