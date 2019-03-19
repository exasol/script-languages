import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.build_or_pull_db_test_image import BuildOrPullDBTestImage
from build_utils.lib.data.dependency_collector.dependency_database_info_collector import DependencyDatabaseInfoCollector
from build_utils.lib.data.dependency_collector.dependency_environment_info_collector import ENVIRONMENT_INFO
from build_utils.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from build_utils.lib.data.environment_info import EnvironmentInfo
from build_utils.lib.test_runner.populate_data import PopulateData
from build_utils.lib.test_runner.spawn_test_container import SpawnTestContainer
from build_utils.lib.test_runner.spawn_test_database import SpawnTestDatabase
from build_utils.lib.test_runner.upload_exa_jdbc import UploadExaJDBC
from build_utils.lib.test_runner.upload_virtual_schema_jdbc_adapter import UploadVirtualSchemaJDBCAdapter


class SpawnTestEnvironment(luigi.Task):
    environment_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._environment_info_target = luigi.LocalTarget(
            "%s/test-runner/db-test/test-environment/%s/environment_info"
            % (self._build_config.ouput_directory,
               self.environment_name))
        if self._environment_info_target.exists():
            self._environment_info_target.remove()

    def requires(self):
        return {
            "database": SpawnTestDatabase(environment_name=self.environment_name),
            "db_test_image": BuildOrPullDBTestImage()
        }

    def output(self):
        return {
            ENVIRONMENT_INFO: self._environment_info_target,
        }

    def run(self):
        database_info_of_dependencies = \
            DependencyDatabaseInfoCollector().get_from_dict_of_inputs(self.input())
        database_info = database_info_of_dependencies["database"]
        database_info_json = database_info.to_json()
        image_info_of_dependencies = \
            DependencyImageInfoCollector().get_from_dict_of_inputs(self.input())
        image_info = image_info_of_dependencies["db_test_image"]
        test_container_info = yield SpawnTestContainer(image_info=image_info,
                                                       database_info=database_info_json,
                                                       container_name=self.environment_name)
        yield [UploadExaJDBC(test_container_info=test_container_info,
                             database_info=database_info_json),
               UploadVirtualSchemaJDBCAdapter(test_container_info=test_container_info,
                                              database_info=database_info_json),
               PopulateData(test_container_info=test_container_info,
                            database_info=database_info)]
        EnvironmentInfo(database_info=database_info, test_container_info=test_container_info)
        with self.output()[ENVIRONMENT_INFO].open("w") as file:
            file.write(database_info.to_json())
