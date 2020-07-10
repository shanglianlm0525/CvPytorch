import warnings
import pathlib

from copy import deepcopy
from collections import UserDict
from abc import abstractclassmethod

import json
import yaml
import pprint

warnings.simplefilter('once', UserWarning)

class Configuration(UserDict):
    def __init__(self, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)

    def __setattr__(self, name, value):
        super(Configuration, self).__setattr__(name, value)
        super(Configuration, self).__setitem__(name, value)

    def __setitem__(self, key, value):
        super(Configuration, self).__setitem__(key, value)
        super(Configuration, self).__setattr__(key, value)

    def sync_attrs_with_data(self):
        for k, v in self.data.items():
            setattr(self, k, v)

    def items(self):
        for k, v in super(Configuration, self).items():
            if k == 'data' or k.startswith('_'):
                continue
            yield k, v

    def keys(self):
        return [k for k, _ in self.items()]

    def raw(self):
        """Return the raw python dict with the same content."""
        d = {}
        for k, v in self.items():
            d[k] = v.raw() if isinstance(v, Configuration) else v
        return d

    def recursive_set(self, name, value):
        d = getattr(self, name)
        if (isinstance(value, UserDict) or isinstance(value, dict)) and isinstance(d, Configuration):
            for k, v in value.items():
                d.recursive_set(k, v)
        else:
            setattr(self, name, value)

    def update(*args, **kwds):
        self, other, *args = args
        for k, v in other.items():
            self.recursive_set(k, v)

    @abstractclassmethod
    def from_dict(cls, d):
        pass

    @abstractclassmethod
    def from_yaml(cls, path):
        pass

    @abstractclassmethod
    def from_json(cls, path):
        pass

    @staticmethod
    def validate_path(path):
        try:
            assert pathlib.Path(path).exists()  # TypeError if path is None; AssertionError if path not exist.
        except (TypeError, AssertionError):
            raise ConfigurationCreationError('Invalid path provided when calling from_yaml. path: {}'.format(path))

    @staticmethod
    def validate_dict(d):
        if d is None or not (isinstance(d, dict) or isinstance(d, Configuration)):
            raise ConfigurationCreationError('Invalid input provided when calling from_dict. input: {}'.format(d))

    def print(self):
        """
        This function will create a copy of self and removes all the 'data' key for better print format.
        """
        _self = deepcopy(self)
        def _pop_data_key(d, r):
            for k, v in r.items():
                if k == 'data' or k.startswith('_'):
                    d.pop(k)
                if isinstance(v, UserDict):
                    _pop_data_key(d[k], v)

        _pop_data_key(_self, self)
        pprint.pprint(_self)


class CommonConfiguration(Configuration):
    def __init__(self, *args, warning_suppress=False, **kwargs):
        super(CommonConfiguration, self).__init__(*args, **kwargs)
        self._warning_suppress = warning_suppress

    def __getattr__(self, item):
        if item.startswith('__') and item.endswith('__'):
            raise AttributeError
        else:
            if not self._warning_suppress:
                msg = 'Config item {} not defined or properly set in config file.'.format(item)
                warnings.warn(msg)

    @classmethod
    def from_yaml(cls, path, warning_suppress=False):
        cls.validate_path(path)
        with open(path, 'r') as f:
            y = yaml.load(f)
        return CommonConfiguration.from_dict(y, warning_suppress=warning_suppress)

    @classmethod
    def from_json(cls, path, warning_suppress=False):
        cls.validate_path(path)
        with open(path, 'r') as f:
            j = json.load(f)
        return CommonConfiguration.from_dict(j, warning_suppress=warning_suppress)

    @classmethod
    def from_dict(cls, d, warning_suppress=False):
        cls.validate_dict(d)
        cfg = CommonConfiguration(warning_suppress=warning_suppress)
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(cfg, k, CommonConfiguration.from_dict(v, warning_suppress=warning_suppress))
            elif isinstance(v, list):
                setattr(cfg, k, [CommonConfiguration.from_dict(d, warning_suppress=warning_suppress)
                                 if isinstance(d, dict) else d for d in v])
            else:
                setattr(cfg, k, v)
        return cfg
