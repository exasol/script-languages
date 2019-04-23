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


def docker_options(login_required: bool):
    return [
        click.option('--docker-base-url', type=str,
                     default="unix:///var/run/docker.sock",
                     show_default=True,
                     help="URL to the socket of the docker daemon."),
        click.option('--docker-repository-name', type=str,
                     default="exasol/script-language-container",
                     show_default=True,
                     help="Name of the docker repository for naming, pushing or fetching cached stages. "
                          "The repository name may contain URL of the docker registory, "
                          "the username and the actual repository name. "
                          "A common strcuture is <docker-registry-url>/<username>/<repository-name>"),
        click.option('--docker-username', type=str,
                     help="Username for the docker registry from where the system pulls cached stages.",
                     required=login_required),
        click.option('--docker-password', type=str,
                     help="Password for the docker registry from where the system pulls cached stages. "
                          "Without password option the system prompts for the password."),
    ]


docker_options_login_not_required = docker_options(login_required=False)

docker_options_login_required = docker_options(login_required=True)

output_directory = click.option('--output-directory', type=click.Path(file_okay=False, dir_okay=True),
                                default=".build_output",
                                show_default=True,
                                help="Output directory where the system stores all output and log files.")
build_options = [
    click.option('--force-build/--no-force-build', default=False,
                 help="Forces the system to complete rebuild of a all stages."),
    click.option('--force-pull/--no-force-pull', default=False,
                 help="Forces the system to pull all stages if available, otherwise it rebuilds a stage."),

    output_directory,
    click.option('--temporary-base-directory',
                 type=click.Path(file_okay=False, dir_okay=True),
                 default="/tmp",
                 show_default=True,
                 help="Directory where the system creates temporary directories."
                 ),
    click.option('--log-build-context-content/--no-log-build-context-content',
                 default=False,
                 help="For Debugging: Logs the files and directories in the build context of a stage"),
]

system_options = [
    click.option('--workers', type=int,
                 default=5, show_default=True,
                 help="Number of parallel workers")
]

release_options = [
    click.option('--release-type',
                 type=click.Choice(['Release', 'BaseTest', "FlavorTest"]),
                 default="Release"
                 )
]
