import configparser
import os


'''
Dump a settings file to the user's home directory

Right now only the DEFAULT section of the config parser is used. If you want
to use a different section make sure to adopt the PytchConfig class.
'''

class PytchConfig():

    def __init__(self):
        config = load_config()

        self.show_traces = config['DEFAULT'].getboolean('show_traces')
        self.start_maximized = config['DEFAULT'].getboolean('start_maximized')
        device_index = config['DEFAULT'].get('device_index')
        if device_index == 'None':
            self.device_index = None
        else:
            device_index = int(device_index)


def load_config():
    '''Read the local configurations file and returned a parsed dictonary.
    If the file does not exist, create a fresh one.'''

    config_file_path = os.path.join(os.getenv("HOME"), '.pytch.config')
    config = configparser.ConfigParser()

    if not os.path.isfile(config_file_path):
        # create config file in home directory with default settings
        config['DEFAULT'] = {
            'device_index': 'None',
            'show_traces': 'False',
            'start_maximized': 'True'}

        with open(config_file_path, 'w') as out:
            config.write(out)

        logger.info('Created new config file in: %s' % config_file_path)

    # parse config file in home directory
    config.read(config_file_path)

    return config


def get_config():
    p = PytchConfig()
    return p
