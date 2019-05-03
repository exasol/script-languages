import jsonpickle
import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.docker_flavor_build_base import DockerFlavorBuildBase
from exaslct_src.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector


class DockerBuild(DockerFlavorBuildBase):
    goals = luigi.ListParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_outputs()

    def get_goals(self):
        return self.goals

    def _prepare_outputs(self):
        self._image_info_target = luigi.LocalTarget(
            "%s/info/image/final"
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
