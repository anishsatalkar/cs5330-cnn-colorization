import json


class ConfigReader(object):
    @staticmethod
    def read():
        with open('config.json') as config_file:
            config = json.load(config_file)
            return config
