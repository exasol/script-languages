import luigi
from luigi import Config

from luigi.parameter import ParameterVisibility


class ExternalDatabaseXMLRPCParameter(Config):
    external_exasol_xmlrpc_host = luigi.OptionalParameter()
    external_exasol_xmlrpc_port = luigi.IntParameter(443)
    external_exasol_xmlrpc_user = luigi.OptionalParameter()
    external_exasol_xmlrpc_cluster_name = luigi.OptionalParameter()
    external_exasol_xmlrpc_password = luigi.OptionalParameter(significant=False,
                                                              visibility=ParameterVisibility.HIDDEN)


class ExternalDatabaseHostParameter(Config):
    external_exasol_db_host = luigi.OptionalParameter()
    external_exasol_db_port = luigi.IntParameter(8563)
    external_exasol_bucketfs_port = luigi.IntParameter(6583)


class ExternalDatabaseCredentialsParameter(ExternalDatabaseHostParameter,
                                           ExternalDatabaseXMLRPCParameter):
    external_exasol_db_user = luigi.OptionalParameter()
    external_exasol_db_password = luigi.OptionalParameter(significant=False,
                                                          visibility=ParameterVisibility.HIDDEN)
    external_exasol_bucketfs_write_password = luigi.OptionalParameter(significant=False,
                                                                      visibility=ParameterVisibility.HIDDEN)
