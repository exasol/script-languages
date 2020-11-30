#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData, expectedFailure
import docker_db_environment

class JavaHive(udf.TestCase):

    def setUp(self):
        self.query('CREATE SCHEMA JAVA_HIVE', ignore_errors=True)
        self.query('OPEN SCHEMA JAVA_HIVE')

    @udf.skipIfNot(docker_db_environment.is_available, reason="This test requires a docker-db environment")
    def test_java_hive(self):
        env=docker_db_environment.DockerDBEnvironment("JAVA_HIVE")
        try:
    #        env.get_client().images.pull("bde2020/hadoop-namenode:2.0.0-hadoop2.7.4-java8")
    #        env.get_client().images.pull("bde2020/hadoop-datanode:2.0.0-hadoop2.7.4-java8")
    #        env.get_client().images.pull("bde2020/hive:2.3.2-postgresql-metastore")
    #        env.get_client().images.pull("bde2020/hive-metastore-postgresql:2.3.0")
    #        env.get_client().images.pull("shawnzhu/prestodb:0.181")
#            env.get_client().volumes.create("namenode")
#            env.get_client().volumes.create("datanode")
            script_dir = os.path.dirname(os.path.realpath(__file__))
            container_name_prefix = env.get_container_name_prefix()
            with open(script_dir+"/hive_resources/hadoop-hive.env") as f:
                environment_file_lines = f.readlines()
                environment_file_lines = [line.format(container_name_prefix=container_name_prefix).rstrip()
                                         for line in environment_file_lines if line.rstrip()!=""]
            namenode=env.run(
                name="namenode",
                image="bde2020/hadoop-namenode:2.0.0-hadoop2.7.4-java8",
#                volumes={'namenode': {'bind': '/hadoop/dfs/name', 'mode': 'rw'}},
                environment=["CLUSTER_NAME=test"]+environment_file_lines,
                ports={"50070/tcp":"50070"},
                )
            datanode=env.run(
                name="datanode",
                image="bde2020/hadoop-datanode:2.0.0-hadoop2.7.4-java8",
#                volumes={'datanode': {'bind': '/hadoop/dfs/data', 'mode': 'rw'}},
                environment=[
                    "SERVICE_PRECONDITION={container_name_prefix}namenode:50070".format(container_name_prefix=container_name_prefix)
                    ]+environment_file_lines,
                ports={"50075/tcp":"50075"},
                )
            hive_server=env.run(
                name="hive-server",
                image="bde2020/hive:2.3.2-postgresql-metastore",
                environment=[
                    "HIVE_CORE_CONF_javax_jdo_option_ConnectionURL=jdbc:postgresql://{container_name_prefix}hive-metastore/metastore".format(container_name_prefix=container_name_prefix),
                    "SERVICE_PRECONDITION={container_name_prefix}hive-metastore:9083".format(container_name_prefix=container_name_prefix)
                    ]+environment_file_lines,
                ports={"10000/tcp":"10000"},
                )

            hive_metastore=env.run(
                name="hive-metastore",
                image="bde2020/hive:2.3.2-postgresql-metastore",
                environment=[
                    "SERVICE_PRECONDITION={container_name_prefix}namenode:50070 {container_name_prefix}datanode:50075 {container_name_prefix}hive-metastore-postgresql:5432".format(container_name_prefix=container_name_prefix)
                    ]+environment_file_lines,
                ports={"9083/tcp":"9083"},
                command="/opt/hive/bin/hive --service metastore"
                )

            hive_metastore_postgresql=env.run(
                name="hive-metastore-postgresql",
                image="bde2020/hive-metastore-postgresql:2.3.0",
                )

            presto_coordinator=env.run(
                name="presto-coordinator",
                image="shawnzhu/prestodb:0.181",
                ports={"8080/tcp":"8080"},
                )
            import time
            time.sleep(30)
           
            from io import BytesIO
            import tarfile
            
            file_like_object = BytesIO()
            tar = tarfile.open(fileobj=file_like_object, mode="w")
            tar.add(script_dir+"/hive_resources/retail.sql", "/hive_resources/retail.sql")
            tar.close()
            hive_server.put_archive("/opt/", file_like_object.getvalue())

            exit_code, output = hive_server.exec_run(cmd="bash -c 'hive -f /opt/hive_resources/retail.sql'")
            print(output)
            if exit_code != 0:
                self.fail("bash -c 'hive -f /opt/hive_resources/retail.sql' failed")

            docker_db_container = env.get_docker_db_container()
            
            delete_command_template = "sed -i '/^.* {container_name}$/d' /etc/hosts"
            append_command_template = "echo '{container_ip} {container_name}' >> /etc/hosts"
            command_template = """bash -c "{delete_command}; {append_command}" """.format(
                                    delete_command=delete_command_template, 
                                    append_command=append_command_template)
            for started_container in env.list_started_containers():
                container_ip=env.get_ip_address_of_container(started_container)
                command = command_template.format(
                    container_ip=env.get_ip_address_of_container(started_container),
                    container_name=started_container.name)
                exit_code, output = docker_db_container.exec_run(cmd=command)
                if exit_code != 0:
                    raise Exception(output)


            docker_db_ip = env.get_ip_address_of_container(docker_db_container)
            upload_url="http://{docker_db_ip}:6583/myudfs/hadoop-etl-udfs-v0.0.1-apache-2.8.5-3.0.0.jar".format(docker_db_ip=docker_db_ip)
            username="w"
            password="write" # TOOD hardcoded
            import requests
            from requests.auth import HTTPBasicAuth
            download_url="https://storage.googleapis.com/exasol-script-languages-extras/hadoop-etl-udfs-v0.0.1-apache-2.8.5-3.0.0.jar"
            r_download = requests.get(download_url, stream=True)
            r_upload = requests.put(upload_url, data=r_download.iter_content(10 * 1024), auth=HTTPBasicAuth(username, password))
            r_download.raise_for_status()
            r_upload.raise_for_status()


            self.query(udf.fixindent("""
	            CREATE OR REPLACE JAVA SET SCRIPT IMPORT_HCAT_TABLE(...) EMITS (...) AS
	            %scriptclass com.exasol.hadoop.scriptclasses.ImportHCatTable;
	            %jar /buckets/bfsdefault/myudfs/hadoop-etl-udfs-v0.0.1-apache-2.8.5-3.0.0.jar;
	            /
	            """))
            self.query(udf.fixindent("""
	            CREATE OR REPLACE JAVA SET SCRIPT IMPORT_HIVE_TABLE_FILES(...) EMITS (...) AS
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
	            %scriptclass com.exasol.hadoop.scriptclasses.HCatTableFiles;
	            %jar /buckets/bfsdefault/myudfs/hadoop-etl-udfs-v0.0.1-apache-2.8.5-3.0.0.jar;
	            /
	            """))

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


            self.query("""
	            IMPORT INTO SALES_POSITIONS
	            FROM SCRIPT IMPORT_HCAT_TABLE WITH
  	              HCAT_DB         = 'retail'
  	              HCAT_TABLE      = 'sales_positions'
  	              HCAT_ADDRESS    = 'thrift://%s:9083'
  	              HCAT_USER       = 'hive'
  	              HDFS_USER       = 'hdfs'
  	              PARALLELISM     = 'nproc()';
	            """%hive_metastore.name)

            result = self.query("SELECT * FROM SALES_POSITIONS LIMIT 10;")
            print(result)



        finally:
            del env
                

if __name__ == '__main__':
    udf.main()

