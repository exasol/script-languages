import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict

import luigi

from exaslct_src.cli.common import set_build_config

from exaslct_src.lib.docker_config import docker_config
from exaslct_src.lib.utils.docker_utils import find_images_by_tag
from exaslct_src.stoppable_task import StoppableTask
from exaslct_src.lib.docker.docker_pull_or_build_image_tasks import DockerPullOrBuildImageTask
from exaslct_src.clean_images import CleanImagesStartingWith

IMAGE_NAME = "exaslc_test"


class DockerPullOrBuildImageTaskTest(unittest.TestCase):

    def setUp(self):
        self.client = docker_config().get_client()
        self.temp_dir = tempfile.mkdtemp()
        set_build_config(False, tuple(), False, False, self.temp_dir, "/tmp/")
        self.clean_images()

    def clean_images(self):
        clean_images_starting_with = CleanImagesStartingWith(starts_with_pattern=IMAGE_NAME)
        clean_images_starting_with._prepare_outputs()
        no_scheduling_errors = luigi.build([clean_images_starting_with],
                                           local_scheduler=True, log_level="INFO", workers=1)
        self.stoppable_task = StoppableTask()
        if self.stoppable_task.failed_target.exists() or not no_scheduling_errors:
            raise Exception("Clean up failed")
        self.client.close()

    def tearDown(self):
        try:
            self.clean_images()
        finally:
            shutil.rmtree(self.temp_dir)

    def test_image_create(self):
        no_scheduling_errors = luigi.build([TestTask()], local_scheduler=True, log_level="INFO")
        if self.stoppable_task.failed_target.exists() or not no_scheduling_errors:
            self.fail("Some task failed")
        images = find_images_by_tag(self.client, lambda tag: tag.startswith(IMAGE_NAME))
        self.assertTrue(len(images) > 0, f"Did not found image {IMAGE_NAME} in list {images}")


class TestTask(DockerPullOrBuildImageTask):
    def get_image_name(self) -> str:
        return IMAGE_NAME

    def get_dockerfile(self) -> str:
        path = Path(__file__).parent.joinpath("resources/Dockerfile")
        return str(path)

    def is_rebuild_requested(self) -> bool:
        return False

    def get_mapping_of_build_files_and_directories(self) -> Dict[str, str]:
        return {}


if __name__ == '__main__':
    unittest.main()
