# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""Setup for pip package."""

from setuptools import find_packages, setup

import pcdiff


def read_requirements_file(filepath):
    with open(filepath) as fin:
        requirements = fin.read()
    return requirements


REQUIRED_PACKAGES = read_requirements_file("requirements-dev.txt")


setup(
    name="pcdiff",
    version=pcdiff.__version__,
    description=("A tools to automatically diff precision between Paddle CPU and other custom device."),
    long_description="",
    url="https://github.com/piDack/PCDiff",
    author="PaddlePaddle Author",
    author_email="",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    data_files=[("pcdiff/configs", ["pcdiff/configs/assign_weight.yaml"])],
    include_data_files=True,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache 2.0",
    keywords=("pcdiff automatically diff precision between paddle cpu and other custom device"),
)
