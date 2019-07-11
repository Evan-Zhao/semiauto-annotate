from typing import overload


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config:
    __metaclass__ = Singleton

    def __init__(self):
        self._config_data = {}

    @staticmethod
    def set_all(kvs):
        _config._config_data = kvs

    @staticmethod
    def set(key, value):
        _config._config_data[key] = value

    @staticmethod
    def get(key, default=None):
        if type(key) is str:
            return _config._config_data.get(key, default)
        iterator = iter(key)
        ret = _config._config_data
        for k in iterator:
            if k not in ret:
                return default
            ret = ret[k]
        return ret


_config = Config()
