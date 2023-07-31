import logging
import sys


class ScanproLogger(logging.Logger):

    verbosity_levels = {0: logging.ERROR,
                        1: logging.INFO,
                        2: logging.DEBUG}

    def __init__(self, verbosity=0):

        # Create custom logger logging all five levels
        super().__init__('Scanpro')
        self.setLevel(logging.INFO)

        # Create stream handler for logging to stdout
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        self.addHandler(h)

        self.set_verbosity(verbosity)

    def set_verbosity(self, verbosity):
        """Set the verbosity level of the logger."""

        if verbosity not in self.verbosity_levels:
            raise ValueError(f"Invalid verbosity level: {verbosity}. Verbosity must be 0, 1 or 2.")

        self.setLevel(self.verbosity_levels[verbosity])
