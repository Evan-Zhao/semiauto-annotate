import os.path as osp
import shutil

import yaml


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config:
    __metaclass__ = Singleton

    def __init__(self):
        self._has_been_set = False
        self._config_data = {}

    @staticmethod
    def set_all(kvs, override=False):
        if _config._has_been_set and not override:
            return
        _config._has_been_set = True
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


def update_dict(target_dict, new_dict, validate_item=None):
    for key, value in new_dict.items():
        if validate_item:
            validate_item(key, value)
        if key not in target_dict:
            print('Skipping unexpected key in config: {}'
                  .format(key))
            continue
        if isinstance(target_dict[key], dict) and \
                isinstance(value, dict):
            update_dict(target_dict[key], value, validate_item=validate_item)
        else:
            target_dict[key] = value


# -----------------------------------------------------------------------------


def get_default_config(config_folder):
    config_file = osp.join(config_folder, 'default_config.yaml')
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # save default config to ~/.labelmerc
    user_config_file = osp.join(osp.expanduser('~'), '.labelmerc')
    if not osp.exists(user_config_file):
        try:
            shutil.copy(config_file, user_config_file)
        except Exception:
            print('Failed to save config: {}'.format(user_config_file))

    return config


def validate_config_item(key, value):
    if key == 'validate_label' and value not in [None, 'exact', 'instance']:
        raise ValueError(
            "Unexpected value for config key 'validate_label': {}"
                .format(value)
        )
    if key == 'labels' and value is not None and len(value) != len(set(value)):
        raise ValueError(
            "Duplicates are detected for config key 'labels': {}".format(value)
        )


def init_default_config(config_folder):
    Config.set_all(get_default_config(config_folder))


def init_config(config_folder, config_from_args=None, config_file=None):
    # Configuration load order:
    #
    #   1. default config (lowest priority)
    #   2. config file passed by command line argument or ~/.labelmerc
    #   3. command line argument (highest priority)

    # 1. default config
    config = get_default_config(config_folder)

    # 2. config from yaml file
    if config_file is not None and osp.exists(config_file):
        with open(config_file) as f:
            user_config = yaml.safe_load(f) or {}
        update_dict(config, user_config, validate_item=validate_config_item)

    # 3. command line argument
    if config_from_args is not None:
        update_dict(config, config_from_args,
                    validate_item=validate_config_item)

    Config.set_all(config, override=True)
