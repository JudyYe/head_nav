import os
from typing import Dict
from yacs.config import CfgNode as CN


def to_lower(x: Dict) -> Dict:
    """
    Convert all dictionary keys to lowercase
    Args:
      x (dict): Input dictionary
    Returns:
      dict: Output dictionary with all keys converted to lowercase
    """
    return {k.lower(): v for k, v in x.items()}


def dataset_config(name='datasets.yaml') -> CN:
    """
    Get dataset config file
    Returns:
      CfgNode: Dataset config as a yacs CfgNode object.
    """
    cfg = CN(new_allowed=True)
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), name)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg
