"""Consistent, professional logging for the whole project."""
import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "loan_prediction") -> logging.Logger:
    """Return a configured logger (idempotent)."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root = logging.getLogger("loan_prediction")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)
