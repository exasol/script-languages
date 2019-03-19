import jsonpickle


class Info:
    def to_json(self):
        jsonpickle.set_preferred_backend('simplejson')
        jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)
        return jsonpickle.encode(self)

    @classmethod
    def from_json(cls, json_string):
        loaded_object = jsonpickle.decode(json_string)
        if not isinstance(loaded_object, cls):
            raise TypeError("Type %s of loaded object does not match %s" % (type(loaded_object), cls))
        return loaded_object