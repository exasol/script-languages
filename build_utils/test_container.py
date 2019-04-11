import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.flavor_task import FlavorWrapperTask, FlavorTask
from build_utils.lib.test_runner.test_runner_db_test_task import TestRunnerDBTestTask
from build_utils.stoppable_task import StoppableTask
from build_utils.export_container import ExportContainer_BaseTest, ExportContainer_FlavorTest, \
    ExportContainer_Release
from build_utils.release_type import ReleaseType


class TestContainer_Release(TestRunnerDBTestTask):
    def get_release_task(self, flavor_path):
        return ExportContainer_Release(flavor_path=flavor_path)

    def get_release_type(self):
        return ReleaseType.Release


class TestContainer_BaseTest(TestRunnerDBTestTask):
    def get_release_task(self, flavor_path):
        return ExportContainer_BaseTest(flavor_path=flavor_path)

    def get_release_type(self):
        return ReleaseType.BaseTest


class TestContainer_FlavorTest(TestRunnerDBTestTask):
    def get_release_task(self, flavor_path):
        return ExportContainer_FlavorTest(flavor_path=flavor_path)

    def get_release_type(self):
        return ReleaseType.FlavorTest


class TestContainer(FlavorTask):
    release_types = luigi.ListParameter(["Release"])
    generic_language_tests = luigi.ListParameter([])
    test_folders = luigi.ListParameter([])
    test_files = luigi.ListParameter([])
    test_restrictions = luigi.ListParameter([])
    languages = luigi.ListParameter([None])
    test_environment_vars = luigi.DictParameter({"TRAVIS": ""})

    test_log_level = luigi.Parameter("critical")
    reuse_database = luigi.BoolParameter(False)
    reuse_uploaded_container = luigi.BoolParameter(False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._prepare_outputs()
        if StoppableTask.failed_target.exists():
            print("removed failed target")
            StoppableTask.failed_target.remove()
        self.actual_release_types = [ReleaseType[release_type] for release_type in self.release_types]

    def requires(self):
        return [self.generate_tasks_for_flavor(flavor_path) for flavor_path in self.actual_flavor_paths]

    def generate_tasks_for_flavor(self, flavor_path):
        result = []
        args = dict(flavor_path=flavor_path,
                    reuse_database=self.reuse_database,
                    reuse_uploaded_container=self.reuse_uploaded_container,
                    generic_language_tests=self.generic_language_tests,
                    test_folders=self.test_folders,
                    test_restrictions=self.test_restrictions,
                    log_level=self.test_log_level,
                    test_environment_vars=self.test_environment_vars,
                    languages=self.languages,
                    test_files=self.test_files)
        if ReleaseType.Release in self.actual_release_types:
            result.append(TestContainer_Release(**args))
        if ReleaseType.BaseTest in self.actual_release_types:
            result.append(TestContainer_BaseTest(**args))
        if ReleaseType.FlavorTest in self.actual_release_types:
            result.append(TestContainer_FlavorTest(**args))
        return result

    def _prepare_outputs(self):
        self._target = luigi.LocalTarget(
            "%s/logs/test-runner/db-test/tests/current"
            % (self._build_config.output_directory))
        if self._target.exists():
            self._target.remove()

    def output(self):
        return self._target

    def run(self):
        with self.output().open("w") as out_file:
            for releases in self.input():
                for in_target in releases:
                    with in_target.open("r") as in_file:
                        out_file.write(in_file.read())
                        out_file.write("\n")
                        out_file.write("=================================================")
                        out_file.write("\n")
