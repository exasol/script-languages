from exaslct_src.abstract_method_exception import AbstractMethodException

DEFAULT_RETURN_OBJECT_NAME = "default"


class AbstractTaskFuture:
    def get_output(self, name: str = DEFAULT_RETURN_OBJECT_NAME):
        raise AbstractMethodException()

    def list_outputs(self):
        raise AbstractMethodException()
