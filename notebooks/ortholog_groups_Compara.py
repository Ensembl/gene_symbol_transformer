# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pathlib

# %%
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px

# %% [markdown]
# ## data

# %%
directory = pathlib.Path("../data/orthologs_Compara")

# %%
data_file_path = directory / "primates_orthologs_one2one_w_perc_id.txt"
data = pd.read_csv(data_file_path, sep="\t")

# %%
data

# %%

# %%
data.nunique()

# %%

# %%
for column in data:
    print(data[column].value_counts())
    print()

# %%

# %% [markdown]
# ## generate ortholog groups from connection graph

# %%
graph = nx.Graph(list(zip(data["gene1_stable_id"], data["gene2_stable_id"])))

# %%
print(graph)

# %%
ortholog_groups = [tuple(c) for c in nx.connected_components(graph)]

# %%
print(len(ortholog_groups))

# %%
og_sizes = pd.DataFrame({"og_size": [len(og) for og in ortholog_groups]})


# %%
og_sizes

# %%
og_sizes.describe()

# %%
fig = px.histogram(og_sizes, x="og_size")
fig.show()

# %% [markdown]
# ## ortholog group samples

# %%
og_graphs = [graph.subgraph(c) for c in nx.connected_components(graph)]

# %%
def draw_og_graph(og_graphs, index, figsize=(12, 12), verbose=False):
    ortholog_group = og_graphs[index]

    if not verbose:
        print(f"og size: {len(ortholog_group)}")
    else:
        print(f"og size: {len(ortholog_group)}", ortholog_group.nodes())

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(ortholog_group, pos=nx.spring_layout(ortholog_group), with_labels=True, ax=ax)


# %%
import random

index = random.choice(range(len(og_graphs)))

# draw_og_graph(og_graphs, index)


# %% [markdown]
# ## strongly connected

# %%
draw_og_graph(og_graphs, index=7414)

# %%
draw_og_graph(og_graphs, index=5652)

# %%
draw_og_graph(og_graphs, index=22217)

# %%
draw_og_graph(og_graphs, index=30957)

# %%
draw_og_graph(og_graphs, index=26499)

# %% [markdown]
# ## weakly connected

# %%
draw_og_graph(og_graphs, index=15914)

# %%
draw_og_graph(og_graphs, index=2769)

# %%
draw_og_graph(og_graphs, index=3250)

# %%
draw_og_graph(og_graphs, index=15307)

# %% [markdown]
# ## branches

# %%
draw_og_graph(og_graphs, index=2735)

# %%
draw_og_graph(og_graphs, index=17520)

# %%
draw_og_graph(og_graphs, index=30872)

# %%
draw_og_graph(og_graphs, index=18292)

# %% [markdown]
# ## very weakly connected

# %%
draw_og_graph(og_graphs, index=21482)

# %%
draw_og_graph(og_graphs, index=10662)

# %%
draw_og_graph(og_graphs, index=31042)

# %% [markdown]
# ## messy

# %%
draw_og_graph(og_graphs, index=16577)

# %%
draw_og_graph(og_graphs, index=687)

# %%
draw_og_graph(og_graphs, index=16869)

# %%
draw_og_graph(og_graphs, index=14081)

# %%
draw_og_graph(og_graphs, index=15999)

# %%
