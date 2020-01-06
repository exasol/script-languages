from collections import Mapping

class FrozenDictToDict():
    def convert(self, obj):
        if isinstance(obj, Mapping):
            return {k: self.convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list,tuple)):
            return [self.convert(o) for o in obj]
        else:
            return obj
