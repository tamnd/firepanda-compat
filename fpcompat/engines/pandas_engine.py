"""pandas, the oracle.

Deliberately dull. The whole value of the oracle is that nothing about it is
surprising, so that when the oracle self test fails the bug is somewhere else.

The one decision here is that frames are loaded fresh per case rather than cached
across a run. Caching would be faster and it would mean a case that mutated a frame
in place changed the input of every case after it, which is the class of bug that
makes a suite's results depend on the order it ran in. Loading from an Arrow IPC file
that is memory mapped is cheap enough that the argument does not need to be had.
"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd
import pyarrow as pa

from fpcompat import corpus


class PandasEngine:
    """The reference implementation."""

    name = "pandas"

    def module(self) -> Any:
        """The pandas module itself, which is what a case expression receives."""
        return pd

    def frame(self, name: str) -> pd.DataFrame:
        """Loads one corpus frame as a DataFrame.

        Args:
            name: The corpus frame name.

        Returns:
            The frame, with a default RangeIndex, which the comparison drops when both
            sides have one.
        """
        return corpus.load(name).to_pandas()

    def versions(self) -> dict[str, str]:
        """The versions that go in the result file.

        Returns:
            pandas exactly, and pyarrow and Python because the answer depends on both.
        """
        return {
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
