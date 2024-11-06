import logging
import warnings
from .vmf_contact.model import vmfContactModule
from .vmf_contact.datasets import DATASET_REGISTRY

_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler())
    _logger.propagate = False



