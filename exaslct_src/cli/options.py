from click._unicodefun import click

flavor_options = [
    click.option('--flavor-path',
                 required=True,
                 multiple=True,
                 type=click.Path(exists=True, file_okay=False, dir_okay=True),
                 help="Path to the directory with the flavor definition. "
                      "The last segment of the path is used as the name of the flavor. "
                      "The option can be repeated with different flavors. "
                      "The system will run the command for each flavor.")
]

docker_repository_options = [
    click.option('--source-docker-repository-name', type=str,
                 default="exasol/script-language-container",
                 show_default=True,
                 help="Name of the docker repository for pulling cached stages. "
                      "The repository name may contain the URL of the docker registry, "
                      "the username and the actual repository name. "
                      "A common structure is <docker-registry-url>/<username>/<repository-name>"),
    click.option('--source-docker-tag-prefix', type=str,
                 default="",
                 show_default=True,
                 help="Prefix for the tags which are used for pulling of cached stages"),
    click.option('--source-docker-username', type=str,
                 help="Username for the docker registry from where the system pulls cached stages.",
                 required=False),
    click.option('--source-docker-password', type=str,
                 help="Password for the docker registry from where the system pulls cached stages. "
                      "Without password option the system prompts for the password."),
    click.option('--target-docker-repository-name', type=str,
                 default="exasol/script-language-container",
                 show_default=True,
                 help="Name of the docker repository for naming and pushing images of stages. "
                      "The repository name may contain the URL of the docker registry, "
                      "the username and the actual repository name. "
                      "A common structure is <docker-registry-url>/<username>/<repository-name>"),
    click.option('--target-docker-tag-prefix', type=str,
                 default="",
                 show_default=True,
                 help="Prefix for the tags which are used for naming and pushing of stages"),
    click.option('--target-docker-username', type=str,
                 help="Username for the docker registry where the system pushes images of stages.",
                 required=False),
    click.option('--target-docker-password', type=str,
                 help="Password for the docker registry where the system pushes images of stages. "
                      "Without password option the system prompts for the password."),
]

simple_docker_repository_options = [
    click.option('--docker-repository-name', type=str,
                 default="exasol/script-language-container",
                 show_default=True,
                 help="Name of the docker repository for naming images. "
                      "The repository name may contain the URL of the docker registry, "
                      "the username and the actual repository name. "
                      "A common structure is <docker-registry-url>/<username>/<repository-name>"),
    click.option('--docker-tag-prefix', type=str,
                 default="",
                 show_default=True,
                 help="Prefix for the tags of the images"),
]

output_directory = click.option('--output-directory', type=click.Path(file_okay=False, dir_okay=True),
                                default=".build_output",
                                show_default=True,
                                help="Output directory where the system stores all output and log files.")

tempory_base_directory = click.option('--temporary-base-directory',
                                      type=click.Path(file_okay=False,
                                                      dir_okay=True),
                                      default="/tmp",
                                      show_default=True,
                                      help="Directory where the system creates temporary directories.")

goal_options = [
    click.option('--goal', multiple=True, type=str,
                 help="Selects which build stage will be build or pushed. "
                      "The system will build also all dependencies of the selected build stage. "
                      "The option can be repeated with different stages. "
                      "The system will than build all these stages and their dependencies."
                 )]

build_options = [
    click.option('--force-rebuild/--no-force-rebuild', default=False,
                 help="Forces the system to complete rebuild all stages down to the stages "
                      "specified with the options --force-rebuild-from."),
    click.option('--force-rebuild-from', multiple=True, type=str,
                 help="If the option --force-rebuild is given, "
                      "this options specifies for which stages and dependent stages system will force a rebuild. "
                      "The option can be repeated with different stages. "
                      "The system will than force the rebuild of these stages and their. dependet stages."
                 ),
    click.option('--force-pull/--no-force-pull', default=False,
                 help="Forces the system to pull all stages if available, otherwise it rebuilds a stage."),
    output_directory,
    tempory_base_directory,
    click.option('--log-build-context-content/--no-log-build-context-content',
                 default=False,
                 help="For Debugging: Logs the files and directories in the build context of a stage"),
    click.option('--cache-directory', default=None, type=click.Path(file_okay=False, dir_okay=True, exists=False),
                 help="Directory from where saved docker images can be loaded"),
    click.option('--build-name', default=None, type=str,
                 help="Name of the build. For example: Repository + CI Build Number"),
]

system_options = [
    click.option('--workers', type=int,
                 default=5, show_default=True,
                 help="Number of parallel workers"),
    click.option('--task-dependencies-dot-file', type=click.Path(file_okay=True),
                 default=None, help="Path where to store the Task Dependency Graph as dot file")
]

release_options = [
    click.option('--release-type',
                 type=click.Choice(['Release', 'BaseTest', "FlavorTest"]),
                 default="Release"
                 )
]
