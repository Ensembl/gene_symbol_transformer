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

# %%
import pathlib
import sys

# %%
import matplotlib.pyplot as plt
import pandas as pd

# %%
figsize = (12, 8)
# figsize = (16, 9)

# %%
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 1000)

# %%
data_directory = pathlib.Path("../data")

# %%
dataset_path = data_directory / "dataset.pickle"
data = pd.read_pickle(dataset_path)

# %%
data.head()

# %%
data.sample(10, random_state=5).sort_index()

# %%
data.info()

# %%

# %%

# %%

# %% [markdown] tags=[]
# ## symbol assignement sources

# %%
data["external_db.db_display_name"].nunique()

# %%
data["external_db.db_display_name"].value_counts()

# %%
symbol_sources = sorted(data["external_db.db_display_name"].unique())
symbol_sources

# %%

# %%

# %%

# %% [markdown]
# ## symbol frequency distribution

# %%
data["Xref_symbol"].nunique()

# %%
data["symbol"].nunique()

# %%

# %%
symbol_counts = data["symbol"].value_counts()
symbol_counts

# %%

# %%
figure = plt.figure()
ax = symbol_counts.hist(figsize=figsize, bins=64)
ax.set(xlabel="sequences per symbol", ylabel="number of symbols")
figure.add_axes(ax)

# %%

# %%
figure = plt.figure()
ax = symbol_counts.hist(figsize=figsize, bins=max(symbol_counts))
ax.set(xlabel="sequences per symbol", ylabel="number of symbols")
figure.add_axes(ax)

# %%

# %%
temp_counts = data[
    data["symbol"].isin(symbol_counts.loc[symbol_counts <= 10 + 1].index)
]["symbol"].value_counts()

figure = plt.figure()
ax = temp_counts.hist(figsize=figsize, bins=10, rwidth=0.7, align="left")
ax.set(xlabel="number of sequences", ylabel="number of symbols")
ax.set(xticks=range(1, 10 + 1), xlim=[1 - 1, 10 + 1])
figure.add_axes(ax)

# %%

# %%

# %%

# %% [markdown]
# ## n most frequent symbols

# %%
symbol_counts

# %%

# %%
num_symbols = 3
symbol_counts[num_symbols - 1 : num_symbols + 1]

# %%

# %%
num_symbols = 100
symbol_counts[num_symbols - 1 : num_symbols + 1]

# %%

# %%
num_symbols = 115
symbol_counts[num_symbols - 1 : num_symbols + 1]

# %%

# %%
num_symbols = 1000
symbol_counts[num_symbols - 1 : num_symbols + 1]

# %%

# %%
num_symbols = 1061
symbol_counts[num_symbols - 1 : num_symbols + 1]

# %%

# %%
min_frequency = 300
symbol_counts[symbol_counts >= min_frequency]

# %%

# %%
min_frequency = 200
symbol_counts[symbol_counts >= min_frequency]

# %%

# %%
min_frequency = 100
symbol_counts[symbol_counts >= min_frequency]

# %%

# %%
min_frequency = 24
symbol_counts[symbol_counts >= min_frequency]

# %%

# %%
min_frequency = 11
symbol_counts[symbol_counts >= min_frequency]

# %%

# %%
min_frequency = 10
symbol_counts[symbol_counts >= min_frequency]

# %%

# %%
min_frequency = 9
symbol_counts[symbol_counts >= min_frequency]

# %%

# %%
min_frequency = 8
symbol_counts[symbol_counts >= min_frequency]

# %%

# %%
min_frequency = 7
symbol_counts[symbol_counts >= min_frequency]

# %%

# %%
min_frequency = 6
symbol_counts[symbol_counts >= min_frequency]

# %%
min_frequency = 5
symbol_counts[symbol_counts >= min_frequency]

# %%

# %%
min_frequency = 4
symbol_counts[symbol_counts >= min_frequency]

# %%

# %%
min_frequency = 3
symbol_counts[symbol_counts >= min_frequency]

# %%

# %%
min_frequency = 2
symbol_counts[symbol_counts >= min_frequency]

# %%

# %%

# %%

# %% [markdown]
# ## clade

# %%
data["clade"].value_counts()

# %%

# %%

# %%

# %%
sys.path.append("..")

from utils import genebuild_clades

# %%
genebuild_clades

# %%
clades = set(genebuild_clades.values())

# %%
clades

# %%

# %%

# %%

# %%

# %%

# %% [markdown] tags=[]
# ## sequence length

# %%
data["sequence_length"] = data["sequence"].str.len()

# %%
data.head()

# %%

# %%
data["sequence_length"].sort_values()

# %%

# %%
data.iloc[data["sequence_length"].sort_values().index[-10:]]

# %%

# %%
figure = plt.figure()
ax = data["sequence_length"].hist(figsize=figsize, bins=1024)
ax.axvline(
    x=round(data["sequence_length"].mean() + 0.5 * data["sequence_length"].std()),
    color="r",
    linewidth=1,
)
ax.set(xlabel="sequence length", ylabel="number of sequences")
figure.add_axes(ax)


# %%

# %%

# %%
figure = plt.figure()
ax = data["sequence_length"].hist(figsize=figsize, bins=max(data["sequence_length"]))
ax.axvline(
    x=round(data["sequence_length"].mean() + 0.5 * data["sequence_length"].std()),
    color="r",
    linewidth=1,
)
ax.set(xlabel="sequence length", ylabel="number of sequences")
ax.set_ylim([None, 6000])
figure.add_axes(ax)


# %%

# %%

# %%
