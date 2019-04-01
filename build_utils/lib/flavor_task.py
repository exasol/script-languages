import pathlib

import luigi


class FlavorTask(luigi.Task):
    flavor_paths = luigi.ListParameter(None)
    flavor_path = luigi.Parameter(None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.flavor_paths is not None:
            self.actual_flavor_paths = self.flavor_paths
        elif self.flavor_path is not None:
            self.actual_flavor_paths = [self.flavor_path]
        else:
            raise luigi.parameter.MissingParameterException("either flavor_paths or flavor_path argument is missing")
        for flavor_path in self.actual_flavor_paths:
            if not pathlib.Path(flavor_path).is_dir():
                raise OSError("Flavor path %s not a directory." % flavor_path)


class FlavorWrapperTask(FlavorTask):

    def complete(self):
        return all(r.complete() for r in luigi.task.flatten(self.requires()))
