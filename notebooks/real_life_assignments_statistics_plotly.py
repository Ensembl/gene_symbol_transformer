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
import plotly.graph_objects as go

from plotly.subplots import make_subplots


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

    figure = make_subplots(specs=[[{"secondary_y": True}]])

    figure.add_trace(
        go.Scatter(
            x=thresholds_list,
            y=num_assignments_list,
            name="number of assignments",
            mode="lines",
        ),
        secondary_y=True,
    )

    if text_title:
        title = comparison_csv_path.stem.split(".")[0]
    else:
        title = comparison_csv_path.stem

    figure.update_layout(
        title_text=title,
        autosize=False,
        width=1200,
        height=600,
    )

    figure.update_xaxes(title_text="threshold probability")

    figure.update_yaxes(title_text="number of assignments", secondary_y=True)

    figure.show()


# %%
text_title = True
# text_title = False

for comparison_csv_path in sorted(pathlib.Path("../../symbols").iterdir()):
    print()
    if text_title:
        print(comparison_csv_path.stem)
    plot_threshold_statistics(comparison_csv_path, text_title=text_title)

# %%
