from os.path import dirname

from labelme_base.utils import init_default_config, Config

# Register config to global position.
# see ./default_config.yaml for valid configuration
config_folder = dirname(__file__)
init_default_config(config_folder)
