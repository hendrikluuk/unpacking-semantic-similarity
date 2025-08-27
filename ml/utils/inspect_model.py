from typing import Type, List
from collections import OrderedDict
import torch

class InspectModel():
    def __init__(self, model:Type[torch.nn.Module]):
        """
          Expect {model} to be an instance of or subclass of torch.nn.Module
        """
        self.model = model
        self.flat = OrderedDict()
        self.nested = {}
        self._map_module(model, 'root')

    def _map_module(self, module:Type[torch.nn.Module], key:str, parent:OrderedDict=None, depth:int=0):
        if parent is None:
            parent = self.nested

        module_key = (key, id(module))
        if key != 'root':
            if module_key in self.flat:
                # make sure we are not referencing the same module
                # multiple times
                return
            #print(f"module_key={module_key}, nesting depth={depth}")
            self.flat[module_key] = module

        for child_key, child_module in module.named_children():
            # ignore a reference to {module} in module.named_children()
            if child_key and (child_key, id(child_module)) != module_key:
                parent[child_key] = {'self': child_module, 'children': OrderedDict(), 'parent': parent}
                self._map_module(child_module, child_key, parent[child_key]['children'], depth+1)

    def find_module(self, name) -> List[torch.nn.Module]:
        """ Find module by name """
        result = []
        for key, item in self.flat.items():
            module_name, _ = key
            if name.lower() == module_name.lower():
                # append module
                result.append(item)
        return result


