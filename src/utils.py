import importlib

def load_object(fn_path: str, kwargs: dict):
    """
    Dynamically load and instantiate an object given a full path and kwargs.

    Args:
        fn_path: full path to the class or function, e.g. 'src.datasets.NestedDataset'
        kwargs: dict of keyword arguments to pass to constructor

    Returns:
        Instantiated object
    """
    module_path, attr_name = fn_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls_or_fn = getattr(module, attr_name)
    return cls_or_fn(**kwargs)
