"""Backward-compatible import shim for the renamed PMC GP module."""

from . import pmc_gp as _pmc_gp
from .pmc_gp import *


def __getattr__(name):
    return getattr(_pmc_gp, name)
