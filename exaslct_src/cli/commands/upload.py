import getpass
from typing import Tuple

from click._unicodefun import click

from exaslct_src.cli.cli import cli
from exaslct_src.cli.common import set_build_config, set_docker_repository_config, run_task, add_options, \
    import_build_steps, set_job_id
from exaslct_src.cli.options \
    import build_options, flavor_options, system_options, release_options, \
    docker_repository_options
from exaslct_src.lib.upload_containers import UploadContainers


@cli.command()
@add_options(flavor_options)
@add_options(release_options)
@click.option('--database-host', type=str,
              required=True)
@click.option('--bucketfs-port', type=int, required=True)
@click.option('--bucketfs-username', type=str, required=True)
@click.option('--bucketfs-password', type=str)
@click.option('--bucketfs-https/--no-bucketfs-https', default=False)
@click.option('--bucketfs-name', type=str, required=True)
@click.option('--bucket-name', type=str, required=True)
@click.option('--path-in-bucket', type=str, required=True)
@click.option('--release-name', type=str, default=None)
@add_options(build_options)
@add_options(docker_repository_options)
@add_options(system_options)
def upload(flavor_path: Tuple[str, ...],
           release_goal: str,
           database_host: str,
           bucketfs_port: int,
           bucketfs_username: str,
           bucketfs_password: str,
           bucketfs_https: bool,
           bucketfs_name: str,
           bucket_name: str,
           path_in_bucket: str,
           release_name: str,
           force_rebuild: bool,
           force_rebuild_from: Tuple[str, ...],
           force_pull: bool,
           output_directory: str,
           temporary_base_directory: str,
           log_build_context_content: bool,
           cache_directory: str,
           build_name: str,
           source_docker_repository_name: str,
           source_docker_tag_prefix: str,
           source_docker_username: str,
           source_docker_password: str,
           target_docker_repository_name: str,
           target_docker_tag_prefix: str,
           target_docker_username: str,
           target_docker_password: str,
           workers: int,
           task_dependencies_dot_file: str):
    """
    This command uploads the whole script language container package of the flavor to the database.
    If the stages or the packaged container do not exists locally, the system will build, pull or
    export them before the upload.
    """
    import_build_steps(flavor_path)
    set_build_config(force_rebuild,
                     force_rebuild_from,
                     force_pull,
                     log_build_context_content,
                     output_directory,
                     temporary_base_directory,
                     cache_directory,
                     build_name)
    set_docker_repository_config(source_docker_password, source_docker_repository_name, source_docker_username,
                                 source_docker_tag_prefix, "source")
    set_docker_repository_config(target_docker_password, target_docker_repository_name, target_docker_username,
                                 target_docker_tag_prefix, "target")
    if bucketfs_password is None:
        bucketfs_password = getpass.getpass(
            "BucketFS Password for BucketFS %s and User %s:" % (bucketfs_name, bucketfs_username))

    set_job_id(UploadContainers.__name__)
    task_creator = lambda: UploadContainers(flavor_paths=list(flavor_path),
                                            release_goals=list([release_goal]),
                                            database_host=database_host,
                                            bucketfs_port=bucketfs_port,
                                            bucketfs_username=bucketfs_username,
                                            bucketfs_password=bucketfs_password,
                                            bucket_name=bucket_name,
                                            path_in_bucket=path_in_bucket,
                                            bucketfs_https=bucketfs_https,
                                            release_name=release_name,
                                            bucketfs_name=bucketfs_name)

    success, task = run_task(task_creator, workers, task_dependencies_dot_file)
    if success:
        with task.command_line_output_target.open("r") as f:
            print(f.read())
    else:
        exit(1)