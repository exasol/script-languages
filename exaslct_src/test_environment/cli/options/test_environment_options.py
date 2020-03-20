import click

test_environment_options = [
    click.option('--environment-type', type=click.Choice(['docker_db', 'external_db']), default="""docker_db""",
                 show_default=True,
                 help="""Environment type for tests."""),
    click.option('--max_start_attempts', type=int, default=2,
                 show_default=True,
                 help="""Maximum start attempts for environment""")

]

docker_db_options = [
    click.option('--docker-db-image-version', type=str, default="""6.2.4-d1""",
                 show_default=True,
                 help="""Docker DB Image Version against which the tests should run."""),
    click.option('--docker-db-image-name', type=str, default="""exasol/docker-db""",
                 show_default=True,
                 help="""Docker DB Image Name against which the tests should run.""")
]

external_db_options = [
    click.option('--external-exasol-db-host', type=str,
                 help="""Host name or IP of external Exasol DB, needs to be set if --environment-type=external_db"""),
    click.option('--external-exasol-db-port', type=str,
                 help="""Database port of external Exasol DB, needs to be set if --environment-type=external_db"""),
    click.option('--external-exasol-bucketfs-port', type=str,
                 help="""Bucketfs port of external Exasol DB, needs to be set if --environment-type=external_db"""),
    click.option('--external-exasol-db-user', type=str,
                 help="""User for external Exasol DB, needs to be set if --environment-type=external_db"""),
    click.option('--external-exasol-db-password', type=str,
                 help="""Database Password for external Exasol DB"""),
    click.option('--external-exasol-bucketfs-write-password', type=str,
                 help="""BucketFS write Password for external Exasol DB"""),
    click.option('--external-exasol-xmlrpc-host', type=str,
                 help="""Hostname for the xmlrpc server"""),
    click.option('--external-exasol-xmlrpc-port', type=int,
                 default="""443""", show_default=True,
                 help="""Port for the xmlrpc server"""),
    click.option('--external-exasol-xmlrpc-user', type=str,
                 default="""admin""", show_default=True,
                 help="""User for the xmlrpc server"""),
    click.option('--external-exasol-xmlrpc-password', type=str,
                 help="""Password for the xmlrpc server"""),
    click.option('--external-exasol-xmlrpc-cluster-name', type=str,
                 default="""cluster1""", show_default=True,
                 help="""Password for the xmlrpc server""")

]