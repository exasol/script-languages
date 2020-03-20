import luigi

from exaslct_src.test_environment.lib.data.environment_type import EnvironmentType
from exaslct_src.test_environment.lib.docker_db_test_environment_parameter import DockerDBTestEnvironmentParameter
from exaslct_src.test_environment.lib.external_test_environment_parameter import ExternalDatabaseCredentialsParameter
from exaslct_src.test_environment.lib.general_spawn_test_environment_parameter import \
    GeneralSpawnTestEnvironmentParameter


class SpawnTestEnvironmentParameter(GeneralSpawnTestEnvironmentParameter,
                                    ExternalDatabaseCredentialsParameter,
                                    DockerDBTestEnvironmentParameter):
    environment_type = luigi.EnumParameter(enum=EnvironmentType)
