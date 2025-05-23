import importlib
import json
import os

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


def load_entity_config(config_path: str = None):
    """
    Load entity configuration from JSON, returning:
      - entities: list of entity types (e.g. ['LAW','CASE',...])
      - bio_labels: list of BIO labels (e.g. ['O','B-LAW','I-LAW',...])
      - outside_tag: string for the outside tag (default 'O')
    """
    # Default path relative to project root
    if config_path is None:
        base = os.path.dirname(os.path.dirname(__file__))  # src/
        config_path = os.path.join(base, 'config', 'entities.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    entities = cfg.get('entities', [])
    prefixes = cfg.get('prefixes', ['B', 'I'])
    outside = cfg.get('outside_tag', 'O')
    # Build BIO label list
    bio_labels = [outside]
    for ent in entities:
        for p in prefixes:
            bio_labels.append(f"{p}-{ent}")
    return entities, bio_labels, outside
