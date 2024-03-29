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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
def plot_threshold_statistics(comparison_csv_path, text_title=False):
    complete_df = pd.read_csv(comparison_csv_path, sep="\t")

    thresholds_list = []
    num_assignments_list = []

    # step: 0.01
    start_a = 0
    end_a = 0.9
    num_values_a = 90 + 1

    # step: 0.001
    start_b = 0.9
    end_b = 1
    num_values_b = 100 + 1

    for threshold in np.concatenate(
        [
            np.linspace(start_a, end_a, num_values_a),
            np.linspace(start_b, end_b, num_values_b),
        ]
    ):
        threshold = round(threshold, ndigits=3)

        df = complete_df.loc[complete_df["probability"] >= threshold]

        num_assignments = len(df)
        if num_assignments == 0:
            continue

        thresholds_list.append(threshold)
        num_assignments_list.append(num_assignments)

    figsize = (16, 9)
    _figure, axis_1 = plt.subplots(figsize=figsize)

    axis_2 = axis_1.twinx()
    axis_2.plot(thresholds_list, num_assignments_list, "b-")

    if not text_title:
        axis_1.set(title=comparison_csv_path.stem)

    axis_1.set(xlabel="threshold probability")

    axis_2.set_ylabel("number of assignments", color="b")

    plt.show()


# %%
text_title = True
# text_title = False

for comparison_csv_path in sorted(pathlib.Path("../../symbols").iterdir()):
    print()
    if text_title:
        print(comparison_csv_path.stem)
    plot_threshold_statistics(comparison_csv_path, text_title=text_title)

# %%
