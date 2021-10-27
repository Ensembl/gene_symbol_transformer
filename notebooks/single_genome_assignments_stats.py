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
# # symbol genome assignments statistics

# %%
import pathlib

from gsc_stats import plot_threshold_statistics, plot_threshold_statistics_no_ground_truth, plot_threshold_statistics_plotly_no_ground_truth


# %%
comparison_csv = ""
comparison_csv_path = pathlib.Path(comparison_csv)

# %%
print(comparison_csv_path.stem)
plot_threshold_statistics(comparison_csv_path, text_title=True)

# %%
print(comparison_csv_path.stem)
plot_threshold_statistics_no_ground_truth(comparison_csv_path, text_title=True)

# %%
print(comparison_csv_path.stem)
plot_threshold_statistics_plotly_no_ground_truth(comparison_csv_path, text_title=True)

# %%
