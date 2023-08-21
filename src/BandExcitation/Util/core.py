import warnings
import time


def add_kwargs(obj, check = True, **kwargs):
    for key, value in kwargs.items():
        if not hasattr(obj, key) and check == True:
            warnings.warn(f"Attribute '{key}' does not exist in the object. Setting it now.")
        setattr(obj, key, value)
        
def inherit_attributes(source_obj, target_obj):
    for key, value in source_obj.__dict__.items():
        setattr(target_obj, key, value)