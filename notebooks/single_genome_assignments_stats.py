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
# # single genome assignments statistics

# %%
import pathlib

from gsc_stats import (
    plot_threshold_statistics,
    plot_threshold_statistics_no_ground_truth,
    plot_threshold_statistics_plotly_no_ground_truth,
)


# %%
symbol_assignments_csv = ""
symbol_assignments_csv_path = pathlib.Path(symbol_assignments_csv)

# %%
# print(symbol_assignments_csv_path.stem)
plot_threshold_statistics_plotly_no_ground_truth(
    symbol_assignments_csv_path, text_title=True
)

# %%
print(symbol_assignments_csv_path.stem)
plot_threshold_statistics_no_ground_truth(symbol_assignments_csv_path, text_title=True)

# %%
# has_comparisons = True
has_comparisons = False

if has_comparisons:
    print(symbol_assignments_csv_path.stem)
    plot_threshold_statistics(symbol_assignments_csv_path, text_title=True)

# %%
