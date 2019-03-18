import pathlib

import luigi


class flavor():

    @classmethod
    def get_name_from_path(cls, path:str):
        path = pathlib.PurePath(path)
        flavor_name = path.name
        return flavor_name