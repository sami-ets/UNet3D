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
import torch
import os

from samitorch.factories.factories import CriterionFactory, ModelFactory, OptimizerFactory, MetricsFactory
from samitorch.factories.enums import *
from samitorch.inputs.datasets import NiftiPatchDataset
from samitorch.inputs.transformers import ToNDTensor
from samitorch.inputs.dataloaders import DataLoader
from samitorch.inputs.utils import sample_collate
from torchvision.transforms import Compose

from UNet3D.factories.parsers import *
from UNet3D.utils.utils import split_dataset


class Initializer(object):

    def __init__(self, path: str):
        self._path = path
        self._logger = logging.getLogger("Initializer")
        self._logger.setLevel(logging.INFO)
        # Logging to console
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(process)d] [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)
        self._logger.propagate = False

    def create_configs(self):
        datasets_configs = UNet3DDatasetConfigurationParserFactory().parse(self._path)
        model_config = UNet3DModelsParserFactory().parse(self._path)
        training_config = TrainingConfigurationParserFactory().parse(self._path)
        variable_config = VariableConfigurationParserFactory().parse(self._path)
        logger_config = LoggerConfigurationParserFactory().parse(self._path)

        return datasets_configs, model_config, training_config, variable_config, logger_config

    def create_metrics(self, training_config):
        dice_metric = training_config.metrics["dice"]

        dice = MetricsFactory().create_metric(Metrics.Dice,
                                              num_classes=dice_metric["num_classes"],
                                              ignore_index=dice_metric["ignore_index"],
                                              average=dice_metric["average"],
                                              reduction=dice_metric["reduction"])

        return dice

    def create_criterions(self, training_config):
        criterion_segmenter = CriterionFactory().create_criterion(training_config.criterion)

        return criterion_segmenter

    def create_models(self, model_config):
        segmenter = ModelFactory().create_model(UNetModels.UNet3D, model_config)

        return segmenter

    def create_optimizers(self, training_config, model):
        optimizer_segmenter = OptimizerFactory().create_optimizer(training_config.optimizer["segmenter"]["type"],
                                                                  model.parameters(),
                                                                  lr=training_config.optimizer["segmenter"]["lr"])
        return optimizer_segmenter

    def create_dataset(self, dataset_config):
        dataset = NiftiPatchDataset(source_dir=dataset_config.path + "/TrainingData/Source",
                                    target_dir=dataset_config.path + "/TrainingData/Target",
                                    patch_shape=dataset_config.training_patch_size,
                                    step=dataset_config.training_patch_step,
                                    transform=Compose([ToNDTensor()]))

        return dataset

    def init_process_group(self, running_config):
        if 'WORLD_SIZE' in os.environ:
            running_config.world_size = int(os.environ['WORLD_SIZE']) > 1

        gpu = 0
        running_config.world_size = 1

        if running_config.is_distributed:
            gpu = running_config.local_rank
            torch.cuda.set_device(gpu)
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='env://')
            running_config.world_size = torch.distributed.get_world_size()
        else:
            torch.cuda.set_device(gpu)

        self._logger.info("Running in {} mode with WORLD_SIZE of {}.".format(
            "distributed" if running_config.is_distributed else "non-distributed",
            running_config.world_size))

    def create_dataloader(self, dataset, batch_size, num_workers, validation_split, is_distributed):
        if is_distributed:
            self._logger.info("Initializing distributed Dataloader.")
            train_dataset, valid_dataset = split_dataset(dataset, validation_split)
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)

            return DataLoader(dataset, shuffle=True, validation_split=validation_split,
                              num_workers=num_workers,
                              batch_size=batch_size,
                              samplers=(train_sampler, valid_sampler),
                              collate_fn=sample_collate)
        else:
            self._logger.info("Initializing Dataloader.")
            return DataLoader(dataset, shuffle=True, validation_split=validation_split,
                              num_workers=num_workers,
                              batch_size=batch_size,
                              collate_fn=sample_collate)
