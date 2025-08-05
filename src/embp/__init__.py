from importlib.util import find_spec
from .embp import EMBP


FLAG_TORCH = find_spec("torch") is not None
__all__ = ["EMBP"]
