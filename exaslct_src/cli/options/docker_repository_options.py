import click

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
