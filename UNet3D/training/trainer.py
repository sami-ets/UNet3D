# -*- coding: utf-8 -*-
# Copyright 2019 Pierre-Luc Delisle. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from UNet3D.training.model_trainer import ModelTrainer


class UNet3DTestTrainer(object):

    def __init__(self, config, callbacks):
        self.UNetTrainer = ModelTrainer(config, callbacks, "SegmentationUNetTrainer")

    def train(self):
        self.UNetTrainer.train()

        return self.UNetTrainer.config.model
