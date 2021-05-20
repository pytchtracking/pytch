import configparser
import os
import logging


logger = logging.getLogger("pytch.config")
"""
Dump a settings file to the user's home directory

Right now only the DEFAULT section of the config parser is used. If you want
to use a different section make sure to adopt the PytchConfig class.
"""


class PytchConfig:
    def __init__(self):
        self.config = load_config()

        self.show_traces = self.config["DEFAULT"].getboolean("show_traces")

        self.start_maximized = self.config["DEFAULT"].getboolean("start_maximized")
        self.accept = self.config["DEFAULT"].getboolean("accept")

        self.device_index = self.config["DEFAULT"].get("device_index")
        try:
            self.device_index = int(self.device_index)
        except ValueError:
            self.device_index = None


def load_config():
    """Read the local configurations file and returned a parsed dictionary.
    If the file does not exist, create a fresh one."""

    config_file_path = os.path.join(os.getenv("HOME", ""), ".pytch.config")
    config = configparser.ConfigParser()

    if not os.path.isfile(config_file_path):
        # create config file in home directory with default settings
        config["DEFAULT"] = {
            "show_traces": False,
            "start_maximized": False,
            "accept": False,
            "device_index": "None",
        }
        with open(config_file_path, "w") as out:
            config.write(out)

        logger.info("Created new config file in: %s", config_file_path)

    # parse config file in home directory
    config.read(config_file_path)
    return config


def get_config():
    p = PytchConfig()
    return p
