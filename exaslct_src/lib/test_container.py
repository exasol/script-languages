import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.flavor_task import FlavorTask
from exaslct_src.lib.stoppable_task import StoppableTask
from exaslct_src.lib.release_type import ReleaseType

from exaslct_src.lib.test_runner.spawn_test_environment_parameter import SpawnTestEnvironmentParameter
from exaslct_src.lib.test_runner.test_runner_db_test_task import TestRunnerDBTestTask, StopTestEnvironment


class TestContainer(FlavorTask, SpawnTestEnvironmentParameter):
    release_types = luigi.ListParameter(["Release"])
    generic_language_tests = luigi.ListParameter([])
    test_folders = luigi.ListParameter([])
    test_files = luigi.ListParameter([])
    test_restrictions = luigi.ListParameter([])
    languages = luigi.ListParameter([None])
    test_environment_vars = luigi.DictParameter({"TRAVIS": ""}, significant=False)

    test_log_level = luigi.Parameter("critical", significant=False)
    reuse_uploaded_container = luigi.BoolParameter(False, significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._prepare_outputs()
        stoppable_task = StoppableTask()
        if stoppable_task.failed_target.exists():
            stoppable_task.failed_target.remove()
        self.actual_release_types = [ReleaseType[release_type] for release_type in self.release_types]

    def requires_tasks(self):
        return [self.generate_tasks_for_flavor(flavor_path, release_type)
                for flavor_path in self.actual_flavor_paths
                for release_type in self.actual_release_types]

    def generate_tasks_for_flavor(self, flavor_path, release_type: ReleaseType):
        args = dict(flavor_path=flavor_path,
                    reuse_uploaded_container=self.reuse_uploaded_container,
                    generic_language_tests=self.generic_language_tests,
                    test_folders=self.test_folders,
                    test_restrictions=self.test_restrictions,
                    log_level=self.test_log_level,
                    test_environment_vars=self.test_environment_vars,
                    languages=self.languages,
                    test_files=self.test_files,
                    release_type=release_type.name,
                    environment_type=self.environment_type,
                    reuse_database_setup=self.reuse_database_setup,
                    reuse_test_container=self.reuse_test_container,
                    docker_db_image_name=self.docker_db_image_name,
                    docker_db_image_version=self.docker_db_image_version,
                    reuse_database=self.reuse_database,
                    database_port_forward=self.database_port_forward,
                    bucketfs_port_forward=self.bucketfs_port_forward,
                    max_start_attempts=self.max_start_attempts,
                    external_exasol_db_host=self.external_exasol_db_host,
                    external_exasol_db_port=self.external_exasol_db_port,
                    external_exasol_bucketfs_port=self.external_exasol_bucketfs_port,
                    external_exasol_db_user=self.external_exasol_db_user,
                    external_exasol_db_password=self.external_exasol_db_password,
                    external_exasol_bucketfs_write_password=self.external_exasol_bucketfs_write_password
                    )
        return TestRunnerDBTestTask(**args)

    def _prepare_outputs(self):
        self._target = luigi.LocalTarget(
            "%s/logs/test-runner/db-test/tests/current"
            % (build_config().output_directory))
        if self._target.exists():
            self._target.remove()

    def output(self):
        return self._target

    def run(self):
        with self.output().open("w") as out_file:
            for release in self.input():
                # for in_target in releases:
                with release.open("r") as in_file:
                    out_file.write(in_file.read())
                    out_file.write("\n")
                    out_file.write("=================================================")
                    out_file.write("\n")
