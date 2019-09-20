from click._unicodefun import click

from exaslct_src.cli.cli import cli
from exaslct_src.cli.common import set_build_config, run_tasks, add_options
from exaslct_src.cli.options \
    import system_options, output_directory, tempory_base_directory
from exaslct_src.lib.test_runner.spawn_test_environment import SpawnTestEnvironmentWithDockerDB


@cli.command()
@click.option('--environment-name', type=str, required=True)
@click.option('--database-port-forward', type=int, required=True)
@click.option('--bucketfs-port-forward', type=int, required=True)
@add_options([output_directory])
@add_options([tempory_base_directory])
@add_options(system_options)
def spawn_test_environment(
        environment_name: str,
        database_port_forward: str,
        bucketfs_port_forward: str,
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
    tasks = lambda: [SpawnTestEnvironmentWithDockerDB(
                                                environment_name=environment_name,
                                                database_port_forward=str(database_port_forward),
                                                bucketfs_port_forward=str(bucketfs_port_forward))]

    run_tasks(tasks, workers, task_dependencies_dot_file)
