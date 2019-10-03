from typing import Dict, Union

from exaslct_src.lib.data.info import Info

class RequiredTaskInfo(Info):
    def __init__(self, module_name:str, class_name:str, params:Dict):
        self.params = params
        self.class_name = class_name
        self.module_name = module_name


class RequiredTaskInfoDict():
    def __init__(self, infos:Dict[str,RequiredTaskInfo]):
        self.infos = infos
