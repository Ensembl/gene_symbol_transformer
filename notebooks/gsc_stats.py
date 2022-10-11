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

    title = "assignments vs probability threshold"
    figure.suptitle(title, fontsize=32)

    exact_matches_genus_included = axis_1.plot(
        threshold_values,
        matching_percentages_include,
        color="green",
        linestyle="-",
        label="exact matches, trained with genus",
    )
    num_assignments_genus_included = axis_2.plot(
        threshold_values,
        num_assignments_include,
        color="blue",
        linestyle="-",
        label="# assignments, trained with genus",
    )

    exact_matches_genus_excluded = axis_1.plot(
        threshold_values,
        matching_percentages_exclude,
        color="limegreen",
        linestyle="--",
        label="exact matches, genus excluded",
    )
    num_assignments_genus_excluded = axis_2.plot(
        threshold_values,
        num_assignments_exclude,
        color="deepskyblue",
        linestyle="--",
        label="# assignments, genus excluded",
    )

    if limit_y_axis:
        axis_1.set_ylim([60, 102])
        axis_2.set_ylim([0, None])

    if text_title:
        axis_1.set(title=include_comparison_csv_path.stem)

    axis_1.set_xlabel("probability threshold", fontsize=32)
    axis_1.set_ylabel("exact matches %", color="green", fontsize=32)
    axis_2.set_ylabel("# assignments", color="blue", fontsize=32)

    threshold = 0.9
    # add vertical line at production threshold
    axis_1.axvline(x=threshold, ymin=0, ymax=0.975, color="black")

    axis_1.annotate(
        threshold,
        xy=(threshold, 65),
        xytext=(threshold - 0.06, 65),
        fontsize=24,
    )

    # exact matches with genus included at threshold
    exact_matches_included = matching_percentages_include[threshold_values.index(0.9)]
    axis_1.annotate(
        f"{exact_matches_included:.2f}",
        xy=(threshold, exact_matches_included),
        xytext=(threshold + 0.01, exact_matches_included + 0.5),
        color="green",
        fontsize=20,
    )

    # exact matches with genus excluded at threshold
    exact_matches_excluded = matching_percentages_exclude[threshold_values.index(0.9)]
    axis_1.annotate(
        f"{exact_matches_excluded:.2f}",
        xy=(threshold, exact_matches_excluded),
        xytext=(threshold + 0.01, exact_matches_excluded - 1.6),
        color="limegreen",
        fontsize=20,
    )

    # number of assignments with genus included at threshold
    num_assignments_included = num_assignments_include[threshold_values.index(0.9)]
    axis_2.annotate(
        num_assignments_included,
        xy=(threshold, num_assignments_included),
        xytext=(threshold - 0.09, num_assignments_included - 1100),
        color="blue",
        fontsize=20,
        # arrowprops=dict(arrowstyle="->", linewidth=2),
    )

    # number of assignments with genus excluded at threshold
    num_assignments_excluded = num_assignments_exclude[threshold_values.index(0.9)]
    axis_2.annotate(
        num_assignments_excluded,
        xy=(threshold, num_assignments_excluded),
        xytext=(threshold - 0.09, num_assignments_excluded - 1300),
        color="deepskyblue",
        fontsize=20,
        # arrowprops=dict(arrowstyle="->", linewidth=2),
    )

    lines = (
        exact_matches_genus_included
        + num_assignments_genus_included
        + exact_matches_genus_excluded
        + num_assignments_genus_excluded
    )
    labels = [line.get_label() for line in lines]
    # axis_1.legend(lines, labels, loc="lower left")
    axis_1.legend(lines, labels)

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
