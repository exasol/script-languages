import jsonpickle
from luigi.parameter import Parameter, ParameterVisibility, _no_value


class JsonPickleParameter(Parameter):

    def __init__(self, cls, default=_no_value, is_global=False, significant=True, description=None,
                 config_path=None, positional=True, always_in_help=False, batch_method=None,
                 visibility=ParameterVisibility.PUBLIC):
        super().__init__(default, is_global, significant, description,
                         config_path, positional, always_in_help,
                         batch_method, visibility)
        self.cls = cls

    def parse(self, s):
        jsonpickle.set_preferred_backend('simplejson')
        loaded_object = jsonpickle.decode(s)
        if not isinstance(loaded_object, self.cls):
            raise TypeError("Type %s of loaded object does not match %s" % (type(loaded_object), self.cls))
        return loaded_object

    def serialize(self, x):
        jsonpickle.set_preferred_backend('simplejson')
        return jsonpickle.encode(x)
