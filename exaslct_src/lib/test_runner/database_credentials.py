import luigi
from luigi import Config
from luigi.parameter import ParameterVisibility


class DatabaseCredentials:
    def __init__(self, db_user: str, db_password: str, bucketfs_write_password: str):
        self.bucketfs_write_password = bucketfs_write_password
        self.db_password = db_password
        self.db_user = db_user


class DatabaseCredentialsParameter(Config):
    db_user = luigi.Parameter()
    db_password = luigi.Parameter(significant=False, visibility=luigi.parameter.ParameterVisibility.HIDDEN)

    # If bucketfs_write_password gets none, luigi might run in a loop with WaitForTestDockerDatabase
    bucketfs_write_password = luigi.Parameter(significant=False, visibility=luigi.parameter.ParameterVisibility.HIDDEN)

    def get_database_credentials(self):
        return DatabaseCredentials(self.db_user, self.db_password, self.bucketfs_write_password)
