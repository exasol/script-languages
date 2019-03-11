import logging
import unittest
from typing import Dict

import luigi

from build_utils.docker_pull_or_build_flavor_image_task import DockerPullOrBuildFlavorImageTask


class Task1(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "udfclient-deps"

    def get_additional_build_directories_mapping(self)->Dict[str,str]:
        return {"ext": "../../ext"}

class Task2(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "language-deps"

    def requires(self):
        return {"udfclient_deps": Task1(flavor_path="resources/test-flavor/", log_build_context_content=True, force_build=True)}


class MyTestCase(unittest.TestCase):
    def test_something(self):
        task = Task2(flavor_path="resources/test-flavor/", log_build_context_content=True, force_build=True)
        luigi.build([task], workers=5, local_scheduler=True, log_level='INFO')


if __name__ == '__main__':
    unittest.main()
