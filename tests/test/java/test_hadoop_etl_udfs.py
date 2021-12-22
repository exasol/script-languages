#!/usr/bin/env python3

import time
import os
from io import BytesIO
import tarfile

from exasol_python_test_framework import udf
from exasol_python_test_framework import docker_db_environment
from exasol_python_test_framework.udf.udf_debug import UdfDebugger


class JavaHive(udf.TestCase):

    def setUp(self):
        self.query('CREATE SCHEMA JAVA_HIVE', ignore_errors=True)
        self.query('OPEN SCHEMA JAVA_HIVE')

    def fill_hadoop_hive_env_template(self, resource_dir, env):
        container_name_prefix = env.get_container_name_prefix()
        with open(resource_dir + "/hadoop-hive.env") as f:
            environment_file_lines = f.readlines()
            environment_file_lines = [line.format(container_name_prefix=container_name_prefix).rstrip()
                                      for line in environment_file_lines if line.rstrip() != ""]
            return environment_file_lines

    def start_namenode(self, env, environment_file_lines):
        namenode = env.run(
            name="namenode",
            image="bde2020/hadoop-namenode:2.0.0-hadoop2.7.4-java8",
            environment=["CLUSTER_NAME=test"] + environment_file_lines,
            ports={"50070/tcp": "50070"},
        )
        return namenode

    def start_datanode(self, env, environment_file_lines, namenode):
        datanode = env.run(
            name="datanode",
            image="bde2020/hadoop-datanode:2.0.0-hadoop2.7.4-java8",
            environment=[
                            "SERVICE_PRECONDITION={namenode}:50070".format(namenode=namenode.name)
                        ] + environment_file_lines,
            ports={"50075/tcp": "50075"},
        )
        return datanode

    def start_hive_server(self, env, environment_file_lines, hive_metastore):
        hive_server = env.run(
            name="hive-server",
            image="bde2020/hive:2.3.2-postgresql-metastore",
            environment=[
                            "HIVE_CORE_CONF_javax_jdo_option_ConnectionURL=jdbc:postgresql://{hive_metastore}/metastore" \
                                .format(hive_metastore=hive_metastore.name),
                            "SERVICE_PRECONDITION={hive_metastore}:9083" \
                                .format(hive_metastore=hive_metastore.name)
                        ] + environment_file_lines,
            ports={"10000/tcp": "10000"},
        )
        return hive_server

    def start_hive_metastore(self, env, environment_file_lines, namenode, datanode, hive_metastore_postgresql):
        hive_metastore = env.run(
            name="hive-metastore",
            image="bde2020/hive:2.3.2-postgresql-metastore",
            environment=[
                            "SERVICE_PRECONDITION={namenode}:50070 {datanode}:50075 {hive_metastore_postgresql}:5432" \
                                .format(
                                namenode=namenode.name,
                                datanode=datanode.name,
                                hive_metastore_postgresql=hive_metastore_postgresql.name
                            )
                        ] + environment_file_lines,
            ports={"9083/tcp": "9083"},
            command="/opt/hive/bin/hive --service metastore"
        )
        return hive_metastore

    def start_hive_metastore_postgresql(self, env):
        hive_metastore_postgresql = env.run(
            name="hive-metastore-postgresql",
            image="bde2020/hive-metastore-postgresql:2.3.0",
        )
        return hive_metastore_postgresql

    def copy_hive_sql_to_hive_server(self, hive_server, resource_dir):
        file_like_object = BytesIO()
        tar = tarfile.open(fileobj=file_like_object, mode="w")
        tar.add(resource_dir + "/retail.sql", "/hive_resources/retail.sql")
        tar.close()
        hive_server.put_archive("/opt/", file_like_object.getvalue())

    def setup_hive_schema(self, hive_server, resource_dir):
        self.copy_hive_sql_to_hive_server(hive_server, resource_dir)
        exit_code, output = hive_server.exec_run(cmd="bash -c 'hive -f /opt/hive_resources/retail.sql'")
        if exit_code != 0:
            self.fail("bash -c 'hive -f /opt/hive_resources/retail.sql' failed, %s" % output)

    def add_containers_to_hosts_file_in_docker_db(self, env):
        docker_db_container = env.get_docker_db_container()
        delete_command_template = "sed -i '/^.* {container_name}$/d' /etc/hosts"
        append_command_template = "echo '{container_ip} {container_name}' >> /etc/hosts"
        command_template = """bash -c "{delete_command}; {append_command}" """.format(
            delete_command=delete_command_template,
            append_command=append_command_template)
        for started_container in env.list_started_containers():
            container_ip = env.get_ip_address_of_container(started_container)
            command = command_template.format(
                container_ip=container_ip,
                container_name=started_container.name)
            exit_code, output = docker_db_container.exec_run(cmd=command)
            if exit_code != 0:
                raise Exception(output)

    def upload_hadoop_etl_udf_jar_to_bucket_fs(self, env):
        docker_db_container = env.get_docker_db_container()
        docker_db_ip = env.get_ip_address_of_container(docker_db_container)
        upload_url = "http://{docker_db_ip}:6583/myudfs/hadoop-etl-udfs-v0.0.1-apache-2.8.5-3.0.0.jar".format(
            docker_db_ip=docker_db_ip)
        username = "w"
        password = "write"  # TOOD hardcoded
        import requests
        from requests.auth import HTTPBasicAuth
        download_url = "https://storage.googleapis.com/exasol-script-languages-extras/hadoop-etl-udfs-v1.0.0-apache-hadoop-2.8.5-hive-2.3.7.jar"
        #        download_url="https://storage.googleapis.com/exasol-script-languages-extras/hadoop-etl-udfs-v0.0.1-apache-2.8.5-3.0.0.jar" # this jar causes https://github.com/exasol/hadoop-etl-udfs/issues/52
        r_download = requests.get(download_url, stream=True)
        r_upload = requests.put(upload_url, data=r_download.iter_content(10 * 1024),
                                auth=HTTPBasicAuth(username, password))
        r_download.raise_for_status()
        r_upload.raise_for_status()

    def create_hadoop_etl_udfs(self):
        self.query(udf.fixindent("""
                CREATE OR REPLACE JAVA SET SCRIPT IMPORT_HCAT_TABLE(...) EMITS (...) AS
                %jvmoption -verbose:jni;
                %scriptclass com.exasol.hadoop.scriptclasses.ImportHCatTable;
                %jar /buckets/bfsdefault/myudfs/hadoop-etl-udfs-v0.0.1-apache-2.8.5-3.0.0.jar;
                /
                """))
        self.query(udf.fixindent("""
                CREATE OR REPLACE JAVA SET SCRIPT IMPORT_HIVE_TABLE_FILES(...) EMITS (...) AS
                %env LD_LIBRARY_PATH=/tmp/;
                %jvmoption -verbose:jni;
                %scriptclass com.exasol.hadoop.scriptclasses.ImportHiveTableFiles;
                %jar /buckets/bfsdefault/myudfs/hadoop-etl-udfs-v0.0.1-apache-2.8.5-3.0.0.jar;
                /
                """))

        self.query(udf.fixindent("""
                CREATE OR REPLACE JAVA SCALAR SCRIPT HCAT_TABLE_FILES(...) EMITS (
                  hdfs_server_port VARCHAR(200),
                  hdfspath VARCHAR(200),
                  hdfs_user_or_service_principal VARCHAR(100),
                  hcat_user_or_service_principal VARCHAR(100),
                  input_format VARCHAR(200),
                  serde VARCHAR(200),
                  column_info VARCHAR(100000),
                  partition_info VARCHAR(10000),
                  serde_props VARCHAR(10000),
                  import_partition INT,
                  auth_type VARCHAR(1000),
                  conn_name VARCHAR(1000),
                  output_columns VARCHAR(100000),
                  enable_rpc_encryption VARCHAR(100),
                  debug_address VARCHAR(200))
                AS
                %jvmoption -verbose:jni;
                %scriptclass com.exasol.hadoop.scriptclasses.HCatTableFiles;
                %jar /buckets/bfsdefault/myudfs/hadoop-etl-udfs-v0.0.1-apache-2.8.5-3.0.0.jar;
                /
                """))

    @udf.skipIfNot(docker_db_environment.is_available, reason="This test requires a docker-db environment")
    def test_java_hive(self):
        env = docker_db_environment.DockerDBEnvironment("JAVA_HIVE")
        try:
            #        env.get_client().images.pull("bde2020/hadoop-namenode:2.0.0-hadoop2.7.4-java8")
            #        env.get_client().images.pull("bde2020/hadoop-datanode:2.0.0-hadoop2.7.4-java8")
            #        env.get_client().images.pull("bde2020/hive:2.3.2-postgresql-metastore")
            #        env.get_client().images.pull("bde2020/hive-metastore-postgresql:2.3.0")
            script_dir = os.path.dirname(os.path.realpath(__file__))
            resource_dir = script_dir + "/resources/hadoop_etl_udf"

            environment_file_lines = self.fill_hadoop_hive_env_template(resource_dir, env)

            namenode = self.start_namenode(env, environment_file_lines)
            datanode = self.start_datanode(env, environment_file_lines, namenode)
            hive_metastore_postgresql = self.start_hive_metastore_postgresql(env)
            hive_metastore = self.start_hive_metastore(env, environment_file_lines, namenode, datanode,
                                                       hive_metastore_postgresql)
            hive_server = self.start_hive_server(env, environment_file_lines, hive_metastore)

            time.sleep(60)  # TODO proper wait strategy for hive

            self.setup_hive_schema(hive_server, resource_dir)
            self.add_containers_to_hosts_file_in_docker_db(env)
            self.upload_hadoop_etl_udf_jar_to_bucket_fs(env)
            self.create_hadoop_etl_udfs()

            self.query("""
                CREATE OR REPLACE TABLE SALES_POSITIONS (
                  SALES_ID    INTEGER,
                  POSITION_ID SMALLINT,
                  ARTICLE_ID  SMALLINT,
                  AMOUNT      SMALLINT,
                  PRICE       DECIMAL(9,2),
                  VOUCHER_ID  SMALLINT,
                  CANCELED    BOOLEAN
                );
                """)

            with UdfDebugger(test_case=self):
                try:
                    self.query("""
                        IMPORT INTO SALES_POSITIONS
                        FROM SCRIPT IMPORT_HCAT_TABLE WITH
                          HCAT_DB         = 'retail'
                          HCAT_TABLE      = 'sales_positions'
                          HCAT_ADDRESS    = 'thrift://%s:9083'
                          HCAT_USER       = 'hive'
                          HDFS_USER       = 'hdfs'
                          PARALLELISM     = 'nproc()';
                        """ % hive_metastore.name)
                finally:
                    print("namenode")
                    print(namenode.logs())
                    print("datanode")
                    print(datanode.logs())
                    print("hive_metastore_postgresql")
                    print(hive_metastore_postgresql.logs())
                    print("hive_metastore")
                    print(hive_metastore.logs())
                    print(hive_server)
                    print(hive_server.logs())
                    time.sleep(10)
        finally:
            del env


if __name__ == '__main__':
    udf.main()
