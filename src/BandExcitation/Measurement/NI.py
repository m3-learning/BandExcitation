import nifgen
import niscope
from nifgen import enums
from niscope import enums as scope_enums
import numpy as np


class FunctionGenerator(nifgen.Session):
    def __init__(self, resource_name, platform="PXI-5413", **kwargs) -> None:
        
        super().__init__(resource_name=resource_name, **kwargs)
        
        self.platform = platform
        
        # makes sure the session is aborted from previous runs
        self.abort()
        
        