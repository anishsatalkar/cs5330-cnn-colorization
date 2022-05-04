import json


class ConfigReader(object):
    @staticmethod
    def read():
        """
        Reads a JSON file that contains the config required for the project.
        :return: Dictionary that contains the config.
        """
        with open('config.json') as config_file:
            config = json.load(config_file)
            return config
