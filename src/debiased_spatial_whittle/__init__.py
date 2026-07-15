__version__ = "2.2.0"

import sys
sys.stdout.isatty = lambda: True
from rich import print
import builtins
builtins.print = print