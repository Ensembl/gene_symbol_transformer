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


"""
Generate groups of orthologs generated by Compara.
"""


# standard library imports

# third party imports
import pandas as pd

# project imports
from utils import data_directory


def save_simplified_orthologs_csv():
    """
    Save a CSV containing just the gene and protein IDs from the original data file.
    """
    original_data_file_path = data_directory / "primates_orthologs_one2one_w_perc_id.txt"
    print(f"loading {original_data_file_path} ...")
    data = pd.read_csv(original_data_file_path, sep="\t")

    columns = ["gene1_stable_id", "protein1_stable_id", "gene2_stable_id", "protein2_stable_id"]
    data = data[columns]

    simplified_orthologs_csv_path = original_data_file_path.parent / "primates_orthologs.csv"
    print(f"saving {simplified_orthologs_csv_path} ...")
    data.to_csv(simplified_orthologs_csv_path, sep="\t", index=False)
    print(f"simplified orthologs CSV saved at {simplified_orthologs_csv_path}")


def main():
    """
    main function
    """
    save_simplified_orthologs_csv()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
