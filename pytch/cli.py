#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Commandline Interface"""

import argparse
import logging

from pytch.gui import start_gui


def main():
    """Parses commandline arguments and starts pytch."""
    parser = argparse.ArgumentParser("pytch")
    parser.add_argument(
        "--debug",
        required=False,
        default=False,
        action="store_true",
        help="Set logging level.",
    )

    args = parser.parse_args()

    logger = logging.getLogger("pytch")
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    start_gui()


# commandline argument parser
if __name__ == "__main__":
    main()
