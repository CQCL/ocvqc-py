"""Emulation of pytket circuits using PECOS"""

from importlib import metadata

__version__ = metadata.version("pytket_pecos")

from .emulator import Emulator
