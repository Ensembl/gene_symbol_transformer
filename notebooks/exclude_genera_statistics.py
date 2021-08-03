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
# # exclude_all_genera

# %%
import pathlib

from gsc_stats import plot_threshold_statistics

# %%
text_title = True
# text_title = False

for comparison_csv_path in sorted(pathlib.Path("../exclude_all_genera").iterdir()):
    print()
    if text_title:
        print(comparison_csv_path.stem)
    plot_threshold_statistics(comparison_csv_path, text_title=text_title)

# %%
