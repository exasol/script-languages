import json

import jsonpickle
import luigi

from exaslct_src.lib.base.frozendict_to_dict import FrozenDictToDict


class Info:
    def to_json(self, indent=4):
        jsonpickle.set_preferred_backend('simplejson')
        jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=indent)
        return jsonpickle.encode(self)

    def to_dict(self):
        return json.loads(self.to_json())

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    @classmethod
    def from_json(cls, json_string):
        loaded_object = jsonpickle.decode(json_string)
        if not isinstance(loaded_object, cls):
            raise TypeError("Type %s of loaded object does not match %s" % (type(loaded_object), cls))
        return loaded_object

    @classmethod
    def from_dict(cls, dictionary):
        converted_dictionary = FrozenDictToDict().convert(dictionary)
        dumps = json.dumps(converted_dictionary)
        return cls.from_json(dumps)
