# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # symbol assignments probability threshold analysis

# %%
import pathlib

from gsc_stats import plot_threshold_statistics_plotly_no_ground_truth

# %%
comparison_csv_path = pathlib.Path("../../assignments/salmo_salar_gca905237065v2_core_104_1_protein_sequences_symbols.csv")

print(comparison_csv_path.stem)
plot_threshold_statistics_plotly_no_ground_truth(comparison_csv_path, text_title=True)

# %%
