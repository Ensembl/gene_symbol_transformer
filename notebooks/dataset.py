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

# %%
import matplotlib.pyplot as plt
import pandas as pd

# %%
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

# %%
data_directory = pathlib.Path("../data")

# %%
dataset_path = data_directory / "dataset.pickle"
data = pd.read_pickle(dataset_path)

# %%
data.head()

# %%
data.sample(10).sort_index()

# %%
data.info()

# %%

# %%

# %%

# %% [markdown] tags=[]
# ## gene symbol assignement sources

# %%
data["external_db.db_display_name"].nunique()

# %%
data["external_db.db_display_name"].value_counts()

# %%
symbol_sources = sorted(data["external_db.db_display_name"].unique())
symbol_sources

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
#figsize = (16, 9)
figsize = (10, 8)

# %%
figure = plt.figure()

ax = symbol_counts.hist(figsize=figsize, bins=32)
ax.set(xlabel="num sequences per symbol", ylabel="num symbols")

figure.add_axes(ax)

#figure.show()

# %%

# %%

# %%
temp_counts = data[data["symbol"].isin(symbol_counts.loc[symbol_counts <= 10].index)]["symbol"].value_counts()

figure = plt.figure()
ax = temp_counts.hist(figsize=(16, 9), bins=10, rwidth=0.6)
ax.set(xlabel="number of sequences", ylabel="number of symbols")
ax.set(xticks=range(1, 10+1), xlim=[0, 11])
figure.add_axes(ax)
#figure.show()

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## n most frequent symbols

# %%
symbol_counts

# %%

# %%
symbol_counts[:3]

# %%
symbol_counts[:4]

# %%

# %%

# %%
symbol_counts[:100]

# %%
symbol_counts[:101]

# %%

# %%
symbol_counts[:1000]

# %%
symbol_counts[:1001]

# %%

# %%
symbol_counts[:1059]

# %%
symbol_counts[:1060]

# %%

# %%
symbol_counts[:25228]

# %%
symbol_counts[:25229]

# %%

# %%
symbol_counts[:30241]

# %%
symbol_counts[:30242]

# %%

# %%

# %%
symbol_counts[:30568]

# %%
symbol_counts[:30569]

# %%

# %%

# %%
symbol_counts[:30911]

# %%
symbol_counts[:30912]

# %%

# %%

# %%
symbol_counts[:31235]

# %%
symbol_counts[:31236]

# %%

# %%

# %%
symbol_counts[:31630]

# %%
symbol_counts[:31631]

# %%

# %%

# %%
symbol_counts[:32068]

# %%
symbol_counts[:32069]

# %%

# %%

# %%
symbol_counts[:32563]

# %%
symbol_counts[:32564]

# %%

# %%

# %%
symbol_counts[:33260]

# %%
symbol_counts[:33261]

# %%

# %%

# %%
symbol_counts[:34461]

# %%
symbol_counts[:34462]

# %%

# %%

# %%
symbol_counts[:37440]

# %%
symbol_counts[:37441]

# %%

# %%

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
import sys
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
