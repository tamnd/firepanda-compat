"""The pandas conformance harness for firepanda.

Nothing here imports pandas at package import time. The modules that need it
import it themselves, so that `python -m fpcompat.report` over a saved result file
works in an environment that has no pandas at all.

The specification is in `docs/specs`, and every module below names the document it
implements.
"""

__version__ = "0.1.0"
