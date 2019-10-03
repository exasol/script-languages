from click._unicodefun import click

from exaslct_src.cli.cli import cli
from exaslct_src.cli.common import set_build_config, run_task, add_options, set_job_id
from exaslct_src.cli.options \
    import system_options, output_directory, tempory_base_directory, docker_db_options
from exaslct_src.lib.test_runner.spawn_test_environment_with_docker_db import SpawnTestEnvironmentWithDockerDB


@cli.command()
@click.option('--environment-name', type=str, required=True)
@click.option('--database-port-forward', type=int, required=True)
@click.option('--bucketfs-port-forward', type=int, required=True)
@add_options(docker_db_options)
@add_options([output_directory])
@add_options([tempory_base_directory])
@add_options(system_options)
def spawn_test_environment(
        environment_name: str,
        database_port_forward: str,
        bucketfs_port_forward: str,
        docker_db_image_version: str,
        docker_db_image_name: str,
        output_directory: str,
        temporary_base_directory: str,
        workers: int,
        task_dependencies_dot_file: str):
    """
    This command spawn a test environment with a docker-db container and a conected test-container.
    The test-container is reachable by the database for output redirects of udfs.
    """
    set_build_config(False,
                     tuple(),
                     False,
                     False,
                     output_directory,
                     temporary_base_directory,
                     None,
                     None)
    task_creator = lambda: SpawnTestEnvironmentWithDockerDB(
        environment_name=environment_name,
        database_port_forward=str(database_port_forward),
        bucketfs_port_forward=str(bucketfs_port_forward),
        docker_db_image_version=docker_db_image_version,
        docker_db_image_name=docker_db_image_name,
        db_user="sys",
        db_password="exasol",
        bucketfs_write_password="write"
    )

    set_job_id(SpawnTestEnvironmentWithDockerDB.__name__)
    success, task = run_task(task_creator, workers, task_dependencies_dot_file)
    if not success:
        exit(1)