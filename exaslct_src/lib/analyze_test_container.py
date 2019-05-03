from typing import Set, Dict

import jsonpickle
import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from exaslct_src.lib.docker.docker_analyze_task import DockerAnalyzeImageTask
from exaslct_src.lib.docker_build_base import DockerBuildBase
from exaslct_src.lib.docker_config import docker_config


class AnalyzeTestContainer(DockerAnalyzeImageTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_image_name(self):
        return f"""{docker_config().repository_name}-db-test-container"""

    def get_image_tag(self):
        return "latest"

    def get_mapping_of_build_files_and_directories(self):
        return {"requirements.txt": "tests/requirements.txt", "ext":"ext"}

    def get_dockerfile(self):
        return "tests/Dockerfile"


class DockerTestContainerBuild(DockerBuildBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actual_flavor_paths=["test-container"] # TODO abtract flavor away
        self._prepare_outputs()

    def get_goal_class_map(self, params)->Dict[str,DockerAnalyzeImageTask]:
        goal_class_map = {"test-container": AnalyzeTestContainer()}
        return goal_class_map

    def get_default_goals(self)->Set[str]:
        goals = {"test-container"}
        return goals

    def get_goals(self):
        goals = {"test-container"}
        return goals

    def _prepare_outputs(self):
        self._image_info_target = luigi.LocalTarget(
            "%s/info/image/test-container"
            % (build_config().output_directory))
        if self._image_info_target.exists():
            self._image_info_target.remove()

    def output(self):
        return self._image_info_target

    def run_task(self):
        tasks_for_all_flavors = self.create_build_tasks_for_all_flavors()
        targets = yield tasks_for_all_flavors
        collector = DependencyImageInfoCollector()
        result = {flavor_path: collector.get_from_dict_of_inputs(image_info_targets)
                  for flavor_path, image_info_targets in targets.items()}
        self.write_output(result)

    def write_output(self, result):
        with self.output().open("w") as f:
            json = self.json_pickle_result(result)
            f.write(json)

    def json_pickle_result(self, result):
        jsonpickle.set_preferred_backend('simplejson')
        jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)
        json = jsonpickle.encode(result)
        return json
