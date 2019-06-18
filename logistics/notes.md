# Project Notes

[Developers] please feel free to put in anything project related!

## Labelme Behavior

### Config Files

- Labelme loads user config from `~/.labelmerc` with high priority, so changes to `default_config.yaml`
may not be effective.

- For consistent behavior, all changes to config are still better committed in default config file, so 
remember to *always remove local config after changing default config* for the change to take effect.
