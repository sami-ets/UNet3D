#  -*- coding: utf-8 -*-
#  Copyright 2019 Pierre-Luc Delisle. All Rights Reserved.
#  #
#  Licensed under the MIT License;
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://opensource.org/licenses/MIT
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import yaml
import logging

from samitorch.configs.configurations import Configuration
from samitorch.factories.parsers import AbstractConfigurationParserFactory
from samitorch.configs.configurations import UNetModelConfiguration

from UNet3D.config.configurations import UNet3DTestDatasetConfiguration, UNet3DTestTrainingConfiguration, \
    VariableConfiguration, LoggerConfiguration


class UNet3DModelsParserFactory(AbstractConfigurationParserFactory):

    def parse(self, path: str):
        with open(path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                unets = UNetModelConfiguration(config["segmenter"])
                return unets
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the config file: {} with error {}".format(path, e))

    def register(self, model_type: str, configuration_class):
        pass


class UNet3DDatasetConfigurationParserFactory(AbstractConfigurationParserFactory):
    def __init__(self) -> None:
        pass

    def parse(self, path: str):
        """
        Parse a dataset configuration file.

        Args:
           path (str): Configuration YAML file path.

        Returns:
           :obj:`samitorch.config.configurations.DatasetConfiguration`: An object containing dataset's properties.

        """
        with open(path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                return UNet3DTestDatasetConfiguration(config["dataset"]["MRBrainS"])
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the config file: {} with error {}".format(path, e))

    def register(self, model_type: str, configuration_class: Configuration):
        pass


class TrainingConfigurationParserFactory(AbstractConfigurationParserFactory):

    def __init__(self):
        pass

    def parse(self, path: str):
        """
        Parse a training configuration file.

        Args:
          path (str): Configuration YAML file path.

        Returns:
          :obj:`samitorch.config.configurations.DatasetConfiguration`: An object containing dataset's properties.

        """
        with open(path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                config = UNet3DTestTrainingConfiguration(config["training"])
                return config
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the config file: {} with error {}".format(path, e))

    def register(self, model_type: str, configuration_class):
        pass


class VariableConfigurationParserFactory(AbstractConfigurationParserFactory):

    def __init__(self):
        pass

    def parse(self, path: str):
        with open(path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                config = VariableConfiguration(config["variables"])
                return config
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the config file: {} with error {}".format(path, e))

    def register(self, model_type: str, configuration_class):
        pass


class LoggerConfigurationParserFactory(AbstractConfigurationParserFactory):

    def __init__(self):
        pass

    def parse(self, path: str):
        with open(path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                config = LoggerConfiguration(config["logger"])
                return config
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the config file: {} with error {}".format(path, e))

    def register(self, model_type: str, configuration_class):
        pass
