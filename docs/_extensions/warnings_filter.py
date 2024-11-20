"""
(taken from https://github.com/zephyrproject-rtos/zephyr/blob/main/doc/_extensions/zephyr/warnings_filter.py)

Warnings filter extension
#########################

Copyright (c) 2021 Nordic Semiconductor ASA
SPDX-License-Identifier: Apache-2.0

Introduction
============

This Sphinx plugin can be used to filter out warnings that are known to be false
positives. The warnings are filtered out based on a set of regular expressions
given via an configuration file. The format of the configuration file is a
plain-text file where each line consists of a regular expression. Any lines
starting with ``#`` will be ignored.

Configuration options
=====================

- ``warnings_filter_config``: Configuration file.
- ``warnings_filter_silent``: Silent flag. If True, warning is hidden. If False
  the warning is converted to an information message and displayed.
"""

import logging
import re
from typing import Dict, Any, List

from sphinx.application import Sphinx
from sphinx.util.logging import NAMESPACE


__version__ = "0.1.0"


class WarningsFilter(logging.Filter):
    """Warnings filter.

    Args:
        expressions: List of regular expressions.
        silent: If true, warning is hidden, otherwise it is shown as INFO.
        name: Filter name.
    """

    def __init__(self, expressions: List[str], silent: bool, name: str = "") -> None:
        super().__init__(name)

        self._expressions = expressions
        self._silent = silent

    def filter(self, record: logging.LogRecord) -> bool:
        print(record)
        print(record.msg)
        print(str(record.msg))
        for expression in self._expressions:
            try:
                if re.match(expression, str(record.msg)):
                    # print("***** WARNINGS-FILTER match*****")
                    if self._silent:
                        # print("***** WARNINGS-FILTER match-silent*****")
                        return False
                    else:
                        # print("***** WARNINGS-FILTER match-loud*****")
                        record.levelno = logging.INFO
                        record.msg = f"Filtered warning: {record.msg}"
                        return True
                elif re.match(".*undefined label: .*", str(record.msg)):
                    print("***** WARNINGS-FILTER difference *****")
                    print(str(record.msg))
            except:
                print("ERROR??", expression, record.msg, type(record.msg))
                raise

        return True


def configure(app: Sphinx) -> None:
    print("***** WARNINGS-FILTER configure *****")
    """Entry point.

    Args:
        app: Sphinx application instance.
    """

    # load expressions from configuration file
    with open(app.config.warnings_filter_config) as f:
        expressions = list()
        for line in f.readlines():
            if not line.startswith("#"):
                expressions.append(line.rstrip())

    # install warnings filter to all the Sphinx logger handlers
    filter = WarningsFilter(expressions, app.config.warnings_filter_silent)
    logger = logging.getLogger(NAMESPACE)
    for handler in logger.handlers:
        handler.filters.insert(0, filter)


def setup(app: Sphinx) -> Dict[str, Any]:
    print("***** WARNINGS-FILTER setup *****")
    app.add_config_value("warnings_filter_config", "", "")
    app.add_config_value("warnings_filter_silent", True, "")

    app.connect("builder-inited", configure)

    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
