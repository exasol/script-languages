#!/usr/bin/env python2.7

import os
import sys
import time 

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
import docker_db_environment

class RCurlSFTPConnectionTest(udf.TestCase):



    @udf.skipIfNot(docker_db_environment.is_available, reason="This test requires a docker-db environment")
    def test_rcurl_sftp_connect(self):
        schema=self.__name__.upper()
        env=docker_db_environment.DockerDBEnvironment(schema)
        try:
            self.query(udf.fixindent("DROP SCHEMA %s CASCADE"%schema),ignore_errors=True)
            self.query(udf.fixindent("CREATE SCHEMA %s"%schema))
            self.query(udf.fixindent("OPEN SCHEMA %s"%schema))
            self.query(udf.fixindent('''
                CREATE OR REPLACE R SCALAR SCRIPT connect_container(
                    host varchar(1000), 
                    port int,
                    username varchar(1000),
                    password varchar(1000), 
                    input_string varchar(1000))  
                    RETURNS VARCHAR(100000) AS

                run <- function(ctx){
                    library(RCurl)
                    file_name = "test_file.txt";
                    file_url_part = paste(ctx$host, "/", file_name, sep = "");
                    upload_url <- paste("ftp://", ctx$username, ":", $ctx.password, "@", file_url_part, sep = "");
                    ftpUpload(what = ctx$input_string, to = upload_url, asText=TRUE);

                    get_url <- paste("ftp://", file_url_part, sep = "");
                    userpwd <- paste(ctx$username, ctx$password, sep = " ");
                    value <- getURL(url, userpwd = userpwd, ftp.use.epsv = FALSE);
                    return(value)
                }
                /
                ''')
            env.get_client().images.pull("panubo/sshd",tag="1.1.0")
            container=env.run(
                name="sshd_sftp",
                image="panubo/sshd:1.1.0",
                environment=["SSH_USERS=test_user:1000:1000","SSH_ENABLE_PASSWORD_AUTH=true","SFTP_MODE=true"],
                tmpfs={'/data': 'size=1M,uid=0'})
            print(container.logs())
            time.sleep(10)
            print(container.logs())
            result=container.exec_run(cmd=''' sh -c "echo 'test_user:test_user' | chpasswd" ''')
            result=container.exec_run(cmd='''mkdir /data/tmp''')
            result=container.exec_run(cmd='''chmod 777 /data/tmp''')
            time.sleep(5)
            print(result)
            host=env.get_ip_address_of_container(container)
            rows=self.query("select connect_container('%s',%s,'test_user','test_user','success')"%(host,22))
            self.assertRowsEqual([("success",)], rows)
            print(container.logs())
        finally:
            try:
                self.query(udf.fixindent("DROP SCHEMA %s CASCADE"%schema))
            except:
                pass
            try:
                env.close()
            except:
                pass


if __name__ == '__main__':
    udf.main()

