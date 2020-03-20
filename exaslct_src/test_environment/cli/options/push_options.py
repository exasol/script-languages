import click

push_options = [
    click.option('--force-push/--no-force-push', default=False,
                 help="Forces the system to overwrite existing images in registry for build steps that run"),
    click.option('--push-all/--no-push-all', default=False,
                 help="Forces the system to push all images of build-steps that are specified by the goals")

]