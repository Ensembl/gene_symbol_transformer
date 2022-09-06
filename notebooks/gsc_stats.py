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


# standard library
import pathlib

# third party imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots


matplotlib.style.use("seaborn-poster")

# figsize = (12, 8)
figsize = (16, 9)


def plot_excluded_clade_statistics(
    include_comparison_csv_path,
    exclude_comparison_csv_path,
    text_title=False,
    limit_y_axis=False,
):
    if not isinstance(include_comparison_csv_path, pathlib.Path):
        include_comparison_csv_path = pathlib.Path(include_comparison_csv_path)
    if not isinstance(exclude_comparison_csv_path, pathlib.Path):
        exclude_comparison_csv_path = pathlib.Path(exclude_comparison_csv_path)

    include_df = pd.read_csv(include_comparison_csv_path, sep="\t")
    exclude_df = pd.read_csv(exclude_comparison_csv_path, sep="\t")

    # step: 0.01
    start_a = 0
    end_a = 0.9
    num_values_a = 90 + 1

    # step: 0.001
    start_b = 0.9
    end_b = 1
    num_values_b = 100 + 1

    threshold_values = [
        round(threshold, ndigits=3)
        for threshold in np.concatenate(
            [
                np.linspace(start_a, end_a, num_values_a),
                np.linspace(start_b, end_b, num_values_b),
            ]
        )
    ]

    num_assignments_include = []
    num_assignments_exclude = []
    matching_percentages_include = []
    matching_percentages_exclude = []
    for threshold in threshold_values:
        num_assignments, matching_percentage = get_threshold_stats(
            include_df, threshold
        )
        num_assignments_include.append(num_assignments)
        matching_percentages_include.append(matching_percentage)

        num_assignments, matching_percentage = get_threshold_stats(
            exclude_df, threshold
        )
        num_assignments_exclude.append(num_assignments)
        matching_percentages_exclude.append(matching_percentage)

    figure, axis_1 = plt.subplots(figsize=figsize)
    axis_1.tick_params(axis="both", which="major", labelsize=24)

    axis_2 = axis_1.twinx()
    axis_2.tick_params(axis="both", which="major", labelsize=24)

    axis_1.plot(
        threshold_values, matching_percentages_include, color="g", linestyle="-"
    )
    axis_2.plot(threshold_values, num_assignments_include, color="b", linestyle="-")

    axis_1.plot(
        threshold_values,
        matching_percentages_exclude,
        color="limegreen",
        linestyle="--",
    )
    axis_2.plot(
        threshold_values, num_assignments_exclude, color="deepskyblue", linestyle="--"
    )

    if limit_y_axis:
        axis_1.set_ylim([60, 102])
        axis_2.set_ylim([0, None])

    if text_title:
        axis_1.set(title=include_comparison_csv_path.stem)

    axis_1.set_xlabel("probability threshold", fontsize=32)
    axis_1.set_ylabel("exact matches %", color="g", fontsize=32)
    axis_2.set_ylabel("# assignments", color="b", fontsize=32)

    plt.show()


def plot_threshold_statistics(
    comparison_csv_path, text_title=False, limit_y_axis=False
):
    complete_df = pd.read_csv(comparison_csv_path, sep="\t")

    threshold_values = []
    num_assignments_list = []
    matching_percentages_list = []

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
        num_exact_matches = len(df.loc[df["exact_match"] == "exact_match"])
        num_fuzzy_matches = len(df.loc[df["fuzzy_match"] == "fuzzy_match"])

        matching_percentage = (num_exact_matches / num_assignments) * 100
        # fuzzy_percentage = (num_fuzzy_matches / num_assignments) * 100
        # num_total_matches = num_exact_matches + num_fuzzy_matches
        # total_matches_percentage = (num_total_matches / num_assignments) * 100

        threshold_values.append(threshold)
        num_assignments_list.append(num_assignments)
        matching_percentages_list.append(matching_percentage)

    figure, axis_1 = plt.subplots(figsize=figsize)
    axis_1.tick_params(axis="both", which="major", labelsize=24)

    axis_2 = axis_1.twinx()
    axis_2.tick_params(axis="both", which="major", labelsize=24)

    axis_1.plot(threshold_values, matching_percentages_list, "g-")
    axis_2.plot(threshold_values, num_assignments_list, "b-")

    if limit_y_axis:
        axis_1.set_ylim([60, 102])
        axis_2.set_ylim([0, None])

    if text_title:
        axis_1.set(title=comparison_csv_path.stem)

    axis_1.set_xlabel("probability threshold", fontsize=32)
    axis_1.set_ylabel("exact matches %", color="g", fontsize=32)
    axis_2.set_ylabel("# assignments", color="b", fontsize=32)

    plt.show()


def plot_threshold_statistics_no_ground_truth(comparison_csv_path, text_title=False):
    complete_df = pd.read_csv(comparison_csv_path, sep="\t")

    threshold_values = []
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

        threshold_values.append(threshold)
        num_assignments_list.append(num_assignments)

    _figure, axis_1 = plt.subplots(figsize=figsize)

    axis_2 = axis_1.twinx()
    axis_2.plot(threshold_values, num_assignments_list, "b-")

    if text_title:
        axis_1.set(title=comparison_csv_path.stem)

    axis_1.set(xlabel="probability threshold")
    axis_1.get_yaxis().set_visible(False)

    axis_2.set_ylabel("number of assignments", color="b")

    plt.show()


def plot_threshold_statistics_plotly(comparison_csv_path, text_title=False):
    complete_df = pd.read_csv(comparison_csv_path, sep="\t")

    threshold_values = []
    num_assignments_list = []
    matching_percentages_list = []

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
        num_exact_matches = len(df.loc[df["exact_match"] == "exact_match"])
        num_fuzzy_matches = len(df.loc[df["fuzzy_match"] == "fuzzy_match"])

        matching_percentage = (num_exact_matches / num_assignments) * 100
        # fuzzy_percentage = (num_fuzzy_matches / num_assignments) * 100
        # num_total_matches = num_exact_matches + num_fuzzy_matches
        # total_matches_percentage = (num_total_matches / num_assignments) * 100

        threshold_values.append(threshold)
        num_assignments_list.append(num_assignments)
        matching_percentages_list.append(matching_percentage)

    figure = make_subplots(specs=[[{"secondary_y": True}]])

    figure.add_trace(
        go.Scatter(
            x=threshold_values,
            y=matching_percentages_list,
            name="exact matches percentage",
            mode="lines",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=threshold_values,
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

    figure.update_yaxes(title_text="exact matches percentage", secondary_y=False)
    figure.update_yaxes(title_text="number of assignments", secondary_y=True)

    figure.show()


def plot_threshold_statistics_plotly_no_ground_truth(
    comparison_csv_path, text_title=False
):
    complete_df = pd.read_csv(comparison_csv_path, sep="\t")

    threshold_values = []
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

        threshold_values.append(threshold)
        num_assignments_list.append(num_assignments)

    figure = make_subplots(specs=[[{"secondary_y": True}]])

    figure.add_trace(
        go.Scatter(
            x=threshold_values,
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
        width=1000,
        height=600,
    )

    figure.update_xaxes(title_text="probability threshold")

    figure.update_yaxes(title_text="number of assignments", secondary_y=True)

    figure.show()


def get_threshold_stats(comparison_df, threshold):
    above_threshold = comparison_df.loc[comparison_df["probability"] >= threshold]

    num_assignments = len(above_threshold)
    num_exact_matches = len(
        above_threshold.loc[above_threshold["exact_match"] == "exact_match"]
    )

    matching_percentage = (num_exact_matches / num_assignments) * 100

    return (num_exact_matches, matching_percentage)
