# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Parquet loading helpers shared by RLVR, OPD, and OPSD datasets."""

import logging

import datasets
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def load_parquet_dataset(parquet_file: str) -> datasets.Dataset:
    """Load a parquet file across Hugging Face ``datasets`` metadata versions.

    Newer ``datasets`` releases serialize list features as ``List``. Older
    releases, including the pinned 3.6.0 reproduction runtime, cannot decode
    that metadata even though PyArrow can read the underlying columns. Fall
    back only for this feature-metadata incompatibility and infer the same
    dataset from the Arrow schema with metadata removed.
    """

    try:
        return datasets.load_dataset("parquet", data_files=parquet_file)["train"]
    except ValueError as error:
        message = str(error)
        if "Feature type '" not in message or "' not found" not in message:
            raise

        logger.warning(
            "Ignoring incompatible Hugging Face feature metadata in %s: %s",
            parquet_file,
            message,
        )
        table = pq.read_table(parquet_file).replace_schema_metadata(None)
        return datasets.Dataset(table)
