from exaslct_src.AbstractMethodException import AbstractMethodException

DEFAULT_RETURN_OBJECT_NAME = "default"


class AbstractTaskFuture:
    def get_output(self, name: str = DEFAULT_RETURN_OBJECT_NAME):
        raise AbstractMethodException()

    def list_outputs(self):
        raise AbstractMethodException()
