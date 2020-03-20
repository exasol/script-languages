from enum import Enum

import jsonpickle


class TaskDescription():

    def __init__(self, id: str, representation: str):
        self.representation = representation
        self.id = id

    def to_json(self):
        jsonpickle.set_preferred_backend('simplejson')
        jsonpickle.set_encoder_options('simplejson', sort_keys=True)
        return jsonpickle.encode(self)

    def __hash__(self):
        return hash(self.to_json())

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.id == other.id and
                self.representation == other.representation
        )

    def __repr__(self):
        return self.id

class DependencyType(Enum):
    requires = 1,
    dynamic = 2


class DependencyState(Enum):
    requested = 1
    finished = 2


class TaskDependency():

    def __init__(self, source: TaskDescription, target: TaskDescription,
                 type: DependencyType, index: int, state: DependencyState):
        self.state = state.name
        self.index = index
        self.type = type.name
        self.target = target
        self.source = source

    def to_json(self):
        jsonpickle.set_preferred_backend('simplejson')
        jsonpickle.set_encoder_options('simplejson', sort_keys=True)
        return jsonpickle.encode(self)

    @classmethod
    def from_json(cls, json_string):
        loaded_object = jsonpickle.decode(json_string)
        if not isinstance(loaded_object, cls):
            raise TypeError("Type %s of loaded object does not match %s" % (type(loaded_object), cls))
        return loaded_object

    def __hash__(self):
        return hash(self.to_json())

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.index == other.index and
                self.state == other.state and
                self.type == other.type and
                self.target == other.target and
                self.source == other.source
        )

    def __repr__(self):
        return f"TaskDependency(source={self.source}, target={self.target}, type={self.type}, index={self.index}, state={self.state})"