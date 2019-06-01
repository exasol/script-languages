import abc
from typing import Dict, List, TypeVar, Generic, Tuple, Set

from luigi import LocalTarget

T = TypeVar('T')

class DependencyInfoCollector(Generic[T]):
    def get_from_dict_of_inputs(self, input: Dict[str, Dict[str, LocalTarget]]) -> Dict[str, T]:
        if isinstance(input, Dict):
            return {key: self.read_info(value)
                    for key, value in input.items()
                    if self.is_info(value)}
        else:
            return dict()

    def get_from_list_of_inputs(self, input: List[Dict[str, LocalTarget]]) -> List[T]:
        if isinstance(input, (List,Tuple,Set)):
            return [self.read_info(value)
                    for value in input
                    if self.is_info(value)]
        else:
            return list()

    def get_from_sinlge_input(self, input: Dict[str, LocalTarget]) -> T:
        if isinstance(input, Dict):
            if self.is_info(input):
                return self.read_info(input)
        return None

    @abc.abstractmethod
    def is_info(self, input):
        pass

    @abc.abstractmethod
    def read_info(self, value)->T:
        pass


