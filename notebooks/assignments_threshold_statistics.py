# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # symbol assignments probability threshold analysis

# %%
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
from gsc_stats import plot_threshold_statistics

# %%
symbol_assignments_directory = ""
symbol_assignments_directory_path = pathlib.Path(symbol_assignments_directory)

# %%
text_title = True
# text_title = False

for comparison_csv_path in sorted(symbol_assignments_directory_path.iterdir()):
    if not str(comparison_csv_path.name).endswith("_compare.csv"):
        continue

    print()
    if text_title:
        print(comparison_csv_path.stem)
    plot_threshold_statistics(comparison_csv_path, text_title=text_title)

# %%
