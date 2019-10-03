from typing import Tuple

from click._unicodefun import click

from exaslct_src.cli.cli import cli
from exaslct_src.cli.common import set_build_config, set_docker_repository_config, run_task, add_options, \
    import_build_steps, set_job_id
from exaslct_src.cli.options \
    import build_options, flavor_options, system_options, docker_repository_options, goal_options
from exaslct_src.lib.docker_build import DockerFlavorBuild, DockerBuild


@cli.command()
@add_options(flavor_options)
@add_options(goal_options)
@add_options(build_options)
@click.option('--shortcut-build/--no-shortcut-build', default=True,
              help="Forces the system to complete to build all all stages, "
                   "but not to rebuild them. If the target images are locally available "
                   "they will be used as is. If the source images locally available "
                   "they will be taged with target image name. "
                   "If the source images can be loaded from file or pulled from a docker registry "
                   "they will get loaded or pulled. The only case, in which them get builded is "
                   "when they are not otherwise available. "
                   "This includes the case where a higher stage which transitivily "
                   "depends on a images is somewhere available, "
                   "but the images as self is not available.")
@add_options(docker_repository_options)
@add_options(system_options)
def build(flavor_path: Tuple[str, ...],
          goal: Tuple[str, ...],
          force_rebuild: bool,
          force_rebuild_from: Tuple[str, ...],
          force_pull: bool,
          shortcut_build: bool,
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
    This command builds all stages of the script language container flavor.
    If stages are cached in a docker registry, they command is going to pull them,
    instead of building them.
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
    set_job_id(DockerBuild.__name__)
    task_creator = lambda: DockerBuild(flavor_paths=list(flavor_path), goals=list(goal), shortcut_build=shortcut_build)
    success, task = run_task(task_creator, workers, task_dependencies_dot_file)
    if not success:
        exit(1)
