from build_utils.lib.data.info import Info


class ContainerInfo(Info):

    def __init__(self, container_name):
        self.container_name = container_name