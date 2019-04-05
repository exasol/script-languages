import luigi

from build_utils.lib.flavor_task import FlavorWrapperTask
from build_utils.lib.test_runner.test_runner_db_test_task import TestRunnerDBTestTask
from build_utils.stoppable_task import StoppableTask
from build_utils.release_container import ReleaseContainer_BaseTest, ReleaseContainer_FlavorTest, \
    ReleaseContainer_Release
from build_utils.release_type import ReleaseType

class TestContainer_Release(TestRunnerDBTestTask):
    def get_release_task(self, flavor_path):
        return ReleaseContainer_Release(flavor_path=flavor_path)

    def get_release_type(self):
        return ReleaseType.Release


class TestContainer_BaseTest(TestRunnerDBTestTask):
    def get_release_task(self, flavor_path):
        return ReleaseContainer_BaseTest(flavor_path=flavor_path)

    def get_release_type(self):
        return ReleaseType.BaseTest


class TestContainer_FlavorTest(TestRunnerDBTestTask):
    def get_release_task(self, flavor_path):
        return ReleaseContainer_FlavorTest(flavor_path=flavor_path)

    def get_release_type(self):
        return ReleaseType.FlavorTest


class TestContainer(FlavorWrapperTask):
    release_types = luigi.ListParameter(["Release"])
    dont_use_flavor_test_config = luigi.BoolParameter(False)
    generic_language_tests = luigi.ListParameter([])
    test_folders = luigi.ListParameter([])
    test_files = luigi.ListParameter([])
    tests_to_execute = luigi.ListParameter([])
    languages = luigi.ListParameter([None])
    environment = luigi.DictParameter({"TRAVIS": ""})

    test_log_level = luigi.Parameter("critical")
    reuse_database = luigi.BoolParameter(False)
    reuse_uploaded_release_container = luigi.BoolParameter(False)
    docker_subnet = luigi.Parameter("12.12.12.0/24")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
                    reuse_uploaded_release_container=self.reuse_uploaded_release_container,
                    generic_language_tests=self.generic_language_tests,
                    test_folders=self.test_folders,
                    tests_to_execute=self.tests_to_execute,
                    log_level=self.test_log_level,
                    environment=self.environment,
                    dont_use_flavor_test_config=self.dont_use_flavor_test_config,
                    languages=self.languages,
                    test_files=self.test_files)
        if ReleaseType.Release in self.actual_release_types:
            result.append(TestContainer_Release(**args))
        if ReleaseType.BaseTest in self.actual_release_types:
            result.append(TestContainer_BaseTest(**args))
        if ReleaseType.FlavorTest in self.actual_release_types:
            result.append(TestContainer_FlavorTest(**args))
        return result
