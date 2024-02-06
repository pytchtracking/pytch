import configparser
import os
import logging


logger = logging.getLogger("pytch.config")
"""
Dump a settings file to the user's home directory

Right now only the DEFAULT section of the config parser is used. If you want
to use a different section make sure to adopt the PytchConfig class.
"""

_color_names = [
    "butter1",
    "butter2",
    "butter3",
    "chameleon1",
    "chameleon2",
    "chameleon3",
    "orange1",
    "orange2",
    "orange3",
    "skyblue1",
    "skyblue2",
    "skyblue3",
    "plum1",
    "plum2",
    "plum3",
    "chocolate1",
    "chocolate2",
    "chocolate3",
    "scarletred1",
    "scarletred2",
    "scarletred3",
    "aluminium1",
    "aluminium2",
    "aluminium3",
    "aluminium4",
    "aluminium5",
    "aluminium6",
    "black",
    "grey",
    "white",
    "red",
    "green",
    "blue",
    "transparent",
]


_color_values = [
    (252, 233, 79),
    (237, 212, 0),
    (196, 160, 0),
    (138, 226, 52),
    (115, 210, 22),
    (78, 154, 6),
    (252, 175, 62),
    (245, 121, 0),
    (206, 92, 0),
    (114, 159, 207),
    (52, 101, 164),
    (32, 74, 135),
    (173, 127, 168),
    (117, 80, 123),
    (92, 53, 102),
    (233, 185, 110),
    (193, 125, 17),
    (143, 89, 2),
    (239, 41, 41),
    (204, 0, 0),
    (164, 0, 0),
    (238, 238, 236),
    (211, 215, 207),
    (186, 189, 182),
    (136, 138, 133),
    (85, 87, 83),
    (46, 52, 54),
    (0, 0, 0),
    (10, 10, 10),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 0, 0, 0),
]


_colors = dict(zip(_color_names, _color_values))


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
            "start_maximized": True,
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
