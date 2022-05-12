#!/usr/bin/env bash


# See the NOTICE file distributed with this work for additional information
# regarding copyright ownership.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# exit on any error
set -e


# install pyenv
# https://github.com/pyenv/pyenv
# curl https://pyenv.run | bash

# install Poetry
# https://github.com/python-poetry/poetry
# curl -sSL https://install.python-poetry.org | python3 -


pyenv install 3.9.10

pyenv virtualenv 3.9.10 gene_symbol_classifier

pip install --upgrade pip

poetry install
