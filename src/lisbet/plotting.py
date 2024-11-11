"""Plotting utilities for LISBET.

This module provides a variety of functions for visualizing data and analysis results
related to the LISBET project. The functions cover a range of tasks, including
plotting UMAP embeddings, F1 score matrices, transition graphs, silhouette profiles,
and dendrograms.

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from scipy.cluster import hierarchy
from scipy.optimize import direct
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.utils.class_weight import compute_class_weight


def mm2inch(x):
    """
    Convert millimeters to inches.

    Parameters
    ----------
    x : float or array-like
        The value(s) in millimeters to convert.

    Returns
    -------
    float or array-like
        The converted value(s) in inches.

    """
    return x / 25.4


def pval2star(pval):
    """
    Convert a p-value to the corresponding star notation (APA style).

    Parameters
    ----------
    pval : float
        The p-value to convert.

    Returns
    -------
    str
        The corresponding star notation, where "***" indicates p less than or equal to
        0.001, "**" indicates p less than or equal to 0.01, "*" indicates p less than or
        equal to 0.05, and "ns" indicates p greater than 0.05.

    """
    star_thresholds = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
    for threshold, star_notation in star_thresholds:
        if pval <= threshold:
            return star_notation
    return "ns"


def get_custom_cmap(n, palette="Set2", alpha=None, desat=None):
    """
    Generate a custom colormap with n colors.

    Parameters
    ----------
    n : int
        Number of colors in the colormap.
    palette : str
        A valid seaborn/matplotlib palette. Default is "Set2".
    alpha : list, optional
        Alpha values for each color in the colormap. If None, no alpha values are
        applied. Default is None.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        A listed colormap with n colors.

    """
    colors = sns.color_palette(palette, desat=desat)

    if n <= len(colors):
        colors = colors[:n]
    else:
        colors = sns.blend_palette(colors, n_colors=n)

    if alpha:
        colors = [c + (a,) for c, a in zip(colors, alpha)]

    # Convert to a *discrete* colormap
    cmap = mpl.colors.ListedColormap(colors)

    return cmap


def plot_umap2d(
    data,
    labels,
    targets=None,
    sample_size=None,
    seed=None,
    marker_size="auto",
    cmap=None,
    cbar_loc="top",
    cbar_label="Motif ID",
    cbar_ticklabels=None,
    ax=None,
):
    """
    Plot a 2D UMAP embedding with class labels.

    Parameters
    ----------
    data : array-like, shape (n_samples, 2)
        The 2D coordinates of the points to plot.
    labels : array-like, shape (n_samples,)
        The class labels for each point.
    sample_size : int, optional
        The number of points to randomly sample for plotting. If None, all points are
        plotted. Default is None.
    seed : int, optional
        Random seed for reproducibility when sampling. Default is None.
    cmap : matplotlib.colors.Colormap, optional
        Colormap for coloring the points. If None, a custom colormap is generated.
        Default is None.
    cbar_loc : str, optional
        Location of the colorbar. Can be "top" or "right". Default is "top".
    cbar_label : str, optional
        Label for the colorbar. Default is "Motif ID".
    cbar_ticklabels : list, optional
        Custom tick labels for the colorbar. If None, labels are based on the unique
        class labels. Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto. If None, a new figure and axes are created.
        Default is None.

    Returns
    -------
    None

    """
    if ax is None:
        _, ax = plt.subplots()

    # Find number of states
    unique_labels = np.unique(labels)
    num_states = len(unique_labels)

    # Sample points, if requested
    if sample_size is not None:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(data.shape[0], replace=False, size=sample_size)
    else:
        sample_idx = ...

    # Compute silhouette score
    sample_silhouette = silhouette_score(data[sample_idx], labels[sample_idx])
    print(f"silhouette_score = {sample_silhouette}")

    # Compute class weights, used to scale marker size in the next scatter plot
    if marker_size == "auto":
        class_weight = dict(
            zip(
                unique_labels,
                compute_class_weight("balanced", classes=unique_labels, y=labels),
            )
        )

        # Scale marker size to compensate for unbalanced classes
        # NOTE: We could use the test labels instead of the predictions to facilitiate
        #       the visual comparison with Figure 1 in the manuscript.
        marker_size = 0.5 * np.array([class_weight[label] for label in labels])
    else:
        marker_size = np.ones_like(labels) * marker_size

    # Assign colors to motifs
    colors = np.array([np.where(unique_labels == label)[0][0] for label in labels])

    if cmap is None:
        cmap = get_custom_cmap(num_states)

    # Plot data
    sc = ax.scatter(
        data[sample_idx, 0],
        data[sample_idx, 1],
        s=marker_size[sample_idx],
        c=colors[sample_idx],
        marker=".",
        vmin=-0.5,
        vmax=num_states - 0.5,
        cmap=cmap,
    )

    if targets is not None:
        mis_idx = np.where(targets != labels)[0]
        mis_sample_idx = np.intersect1d(mis_idx, sample_idx)
        er = ax.scatter(
            data[mis_sample_idx, 0],
            data[mis_sample_idx, 1],
            s=5 * marker_size[mis_sample_idx],
            c=colors[mis_sample_idx],
            marker="x",
            vmin=-0.5,
            vmax=num_states - 0.5,
            cmap=cmap,
        )

    # Create colorbar
    cbar = plt.colorbar(
        sc,
        ax=ax,
        location=cbar_loc,
        ticks=mticker.FixedLocator(range(num_states)),
        label=cbar_label,
        shrink=0.95,
    )

    # Customize ticklabels
    if cbar_ticklabels is None:
        cbar.ax.set_xticklabels(unique_labels)
    else:
        if cbar_loc == "right":
            cbar.ax.set_yticklabels(
                cbar_ticklabels, rotation=90, verticalalignment="center"
            )
        elif cbar_loc == "top":
            cbar.ax.set_xticklabels(
                cbar_ticklabels, rotation=30, horizontalalignment="left"
            )
        else:
            raise NotImplementedError(
                "Only 'top' and 'right' are valid cbar locations."
            )

    # Finalize plot
    ax.set_xlabel("UMAP$_1$")
    ax.set_ylabel("UMAP$_2$")


def plot_f1_matrix(f1_matrix, xlabels=None, ylabels=None, ax=None):
    """
    Plot an F1 score matrix as a heatmap.

    Parameters
    ----------
    f1_matrix : array-like, shape (n_classes, n_classes)
        The F1 score matrix.
    xlabels : list, optional
        Labels for the x-axis. If None, indices are used. Default is None.
    ylabels : list, optional
        Labels for the y-axis. If None, indices are used. Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto. If None, a new figure and axes are created.
        Default is None.

    Returns
    -------
    None
        The plot is drawn on the provided or created axes.

    """
    if ax is None:
        _, ax = plt.subplots()

    if xlabels is None:
        xlabels = range(f1_matrix.shape[1])

    if ylabels is None:
        ylabels = range(f1_matrix.shape[0])

    data = pd.DataFrame(f1_matrix, columns=xlabels, index=ylabels)

    sns.heatmap(
        data,
        annot=True,
        fmt=".1f",
        linewidth=0.5,
        cmap="Blues",
        cbar_kws={"location": "top", "label": "F1 score"},
        ax=ax,
    )

    # Finalize plot
    ax.set_ylabel("Motif ID")
    ax.tick_params(left=False, bottom=False)


def plot_transition_graph(
    trans_prob, node_sizes=150, edge_vmin=None, edge_vmax=None, cmap=None, ax=None
):
    """
    Plots a transition graph based on the provided transition probability matrix.

    Parameters
    ----------
    trans_prob : numpy.ndarray
        A square matrix where element [i, j] represents the probability of transitioning
        from state i to state j.
    node_sizes : int or list, optional
        The size of the nodes in the graph. If an int, all nodes will have the same
        size.
    edge_vmin : float, optional
        Minimum value for the edge colormap normalization.
    edge_vmax : float, optional
        Maximum value for the edge colormap normalization.
    cmap : matplotlib.colors.Colormap, optional
        Colormap for the edges of the graph.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto, otherwise uses the current axes.

    Returns
    -------
    nodes : matplotlib.collections.PathCollection
        The drawn nodes of the graph.
    labels : dict
        The labels of the nodes in the graph.
    cbar : matplotlib.colorbar.Colorbar
        Colorbar corresponding to the edge weights.

    """
    if ax is None:
        _, ax = plt.subplots()

    # Find number of states
    num_states = trans_prob.shape[0]

    # Make graph
    G = nx.from_numpy_array(trans_prob, create_using=nx.DiGraph)
    pos = nx.circular_layout(G)

    edgelist, weights = zip(*nx.get_edge_attributes(G, "weight").items())

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=range(num_states),
        cmap=get_custom_cmap(num_states),
    )

    labels = nx.draw_networkx_labels(
        G,
        pos=pos,
        font_color="w",
        font_size=mpl.rcParams["font.size"],
        font_weight="bold",
        ax=ax,
    )

    edges = nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=10,
        connectionstyle="arc3,rad=0.3",
        edge_cmap=cmap,
        width=2,
        edgelist=edgelist,
        edge_color=weights,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
    )

    # Make colorbar
    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(weights)
    cbar = plt.colorbar(
        pc,
        ax=ax,
        location="top",
        label="Transition probability",
        shrink=0.95,
    )

    # Remove axis
    ax.set_axis_off()

    return nodes, labels, cbar


def plot_points(data, y, ax):
    """
    Plots boxplots and strip plots for the given data based on Motif ID.

    Parameters
    ----------
    data : pandas.DataFrame
        The data containing 'Motif ID' and the variable to plot on the y-axis.
    y : str
        The name of the column in `data` to be plotted on the y-axis.
    ax : matplotlib.axes.Axes
        Axes object to draw the plot onto.

    Returns
    -------
    None

    """
    # Select colors
    num_states = data["Motif ID"].max() + 1
    cmap = get_custom_cmap(num_states)
    palette = [cmap(c) for c in range(num_states)]

    sns.boxplot(
        data=data,
        x="Motif ID",
        y=y,
        hue="Motif ID",
        fill=False,
        showfliers=False,
        legend=False,
        palette=palette,
        ax=ax,
    )

    sns.stripplot(
        data=data,
        x="Motif ID",
        y=y,
        hue="Motif ID",
        size=1.5,
        legend=False,
        palette=palette,
        ax=ax,
    )


def plot_group_points(data, y, test_dict, ax):
    """
    Plots group-wise boxplots and strip plots with statistical test results.

    Parameters
    ----------
    data : pandas.DataFrame
        The data containing 'Motif ID', 'Group label', and the variable to plot on the
        y-axis.
    y : str
        The name of the column in `data` to be plotted on the y-axis.
    test_dict : dict
        A dictionary where keys are Motif IDs and values are statistical test results
        (e.g., t-test results).
    ax : matplotlib.axes.Axes
        Axes object to draw the plot onto.

    Returns
    -------
    None

    """
    sns.boxplot(
        data=data,
        x="Motif ID",
        y=y,
        hue="Group label",
        fill=False,
        showfliers=False,
        ax=ax,
    )

    sns.stripplot(
        data=data,
        x="Motif ID",
        y=y,
        hue="Group label",
        size=1.5,
        dodge=True,
        legend=False,
        ax=ax,
    )

    # Plot pval
    for pos, (motif_id, ttest_result) in enumerate(test_dict.items()):
        print(f"Motif {motif_id} pval = {ttest_result.pvalue:.5f}")
        ax.text(
            pos,
            1,
            pval2star(ttest_result.pvalue),
            horizontalalignment="center",
            verticalalignment="baseline",
            transform=blended_transform_factory(ax.transData, ax.transAxes),
            color="k",
        )


def plot_embedding_summary(
    embeddings,
    labels,
    predictions,
    fps=None,
    cmap_human=None,
    cmap_machine=None,
    axarr=None,
):
    """
    Plots a summary of embeddings over time, including expert and predicted labels.

    Parameters
    ----------
    embeddings : numpy.ndarray
        The embedding matrix where each row corresponds to a time point.
    labels : numpy.ndarray
        Array of expert-provided labels corresponding to the embeddings.
    predictions : numpy.ndarray
        Array of model-predicted labels corresponding to the embeddings.
    axarr : numpy.ndarray of matplotlib.axes.Axes, optional
        Array of Axes objects to draw the plots onto. If not provided, new subplots will
        be created.

    Returns
    -------
    None

    """
    if axarr is None:
        fig, axarr = plt.subplots(
            3,
            2,
            sharex="col",
            height_ratios=(1, 0.2, 0.2),
            width_ratios=(1, 0.02),
            gridspec_kw={"hspace": 0.05},
        )
    else:
        fig = axarr[0][0].get_figure()

    n_classes = int(np.max(labels)) + 1
    n_states = int(np.max(predictions)) + 1

    if fps is not None:
        extent = [0, embeddings.shape[0] / fps, 0, embeddings.shape[1]]
    else:
        extent = None

    # Plot embedding over time
    ax, cax = axarr[0]
    im = ax.imshow(
        embeddings.T,
        aspect="auto",
        interpolation="none",
        cmap="Spectral_r",
        extent=extent,
    )
    fig.colorbar(im, cax=cax, label="Activation")
    ax.set_ylabel("Embedding")
    ax.set_yticks([])

    # Manual annotations
    ax, cax = axarr[1]
    im = ax.imshow(
        labels[np.newaxis],
        aspect="auto",
        interpolation="none",
        cmap=get_custom_cmap(n_classes) if cmap_human is None else cmap_human,
        vmin=-0.5,
        vmax=n_classes - 0.5,
        extent=extent,
    )
    cbar = fig.colorbar(im, cax=cax, ticks=mticker.FixedLocator(range(n_classes)))
    ax.set_ylabel("Human")
    ax.set_yticks([])

    # HMM annotations
    ax, cax = axarr[2]
    im = ax.imshow(
        predictions[np.newaxis],
        aspect="auto",
        interpolation="none",
        cmap=get_custom_cmap(n_states) if cmap_machine is None else cmap_machine,
        vmin=-0.5,
        vmax=n_states - 0.5,
        extent=extent,
    )
    cbar = fig.colorbar(im, cax=cax, label="Motif ID")
    ax.set_ylabel("LISBET")
    ax.set_yticks([])

    # Finalize plot
    fig.align_ylabels(axarr[[0, 2], 1])

    ax.set_xlabel("Frame" if fps is None else "Time (s)")


def plot_slh_score(all_n_clusters, all_score, best_n_clusters, best_score, ax):
    """
    Plots the silhouette score as a function of the number of clusters.

    Parameters
    ----------
    all_n_clusters : list or numpy.ndarray
        A list of different numbers of clusters.
    all_score : list or numpy.ndarray
        Corresponding silhouette scores for each number of clusters.
    best_n_clusters : int
        The number of clusters with the best silhouette score.
    best_score : float
        The best silhouette score obtained.
    ax : matplotlib.axes.Axes
        Axes object to draw the plot onto.

    Returns
    -------
    None

    """
    ax.plot(all_n_clusters, all_score)

    ax.scatter(best_n_clusters, best_score, c="red", label="best")
    ax.axhline(best_score, ls="dashed", color="k")
    # ax.axvline(best_n_clusters, ls="dashed", color="k")

    # Finalize plot
    ax.set_xlabel("No. clusters")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Average")
    ax.legend()


def plot_slh_profile(distance, link_matrix, cluster_labels, ax):
    """
    Plots the silhouette score profile for hierarchical clustering.

    Parameters
    ----------
    distance : numpy.ndarray
        Precomputed distance matrix.
    link_matrix : numpy.ndarray
        Linkage matrix obtained from hierarchical clustering.
    cluster_labels : numpy.ndarray
        Cluster labels for each sample.
    ax : matplotlib.axes.Axes
        Axes object to draw the plot onto.

    Returns
    -------
    None

    """
    n_clusters = len(np.unique(cluster_labels))
    indices = hierarchy.leaves_list(link_matrix)

    # Plot average silhouette score
    slh_avg = silhouette_score(distance, cluster_labels, metric="precomputed")
    ax.axhline(slh_avg, ls="dashed", color="k", label="Average silhouette score")

    # Compute silhouette score profile
    sample_slh_values = silhouette_samples(
        distance, cluster_labels, metric="precomputed"
    )

    # Compute colors
    cmap = get_custom_cmap(n_clusters)

    # Plot silhouette score profile
    x = np.arange(len(sample_slh_values))
    colors = [cmap.colors[cluster_labels[i]] for i in indices]
    ax.bar(x, sample_slh_values[indices], color=colors)

    # Create axes for colobar
    ax_cbar = ax.inset_axes([0, -0.2, 1, 0.15])
    ax_cbar.sharex(ax)

    # Plot colorbar
    cbar_values = np.array(cluster_labels, ndmin=2)
    ax_cbar.matshow(cbar_values[:, indices], aspect="auto", cmap=cmap)

    # Plot cluster ids
    res, ind = np.unique(cluster_labels[indices], return_index=True)
    loc = 0
    for cluster_id in res[np.argsort(ind)]:
        cluster_size = np.sum(cluster_labels == cluster_id)
        if cluster_id % 1 == 0:
            ax_cbar.text(
                loc + cluster_size / 2 - 0.5,
                0,
                f"{cluster_id}",
                horizontalalignment="center",
                verticalalignment="center",
                color="w",
                weight="bold",
            )
        loc += cluster_size

    # Finalize plot
    ax.set_title("Profile")
    ax.set_xticks(range(len(indices)), labels=indices)
    plt.setp(ax.get_yticklabels(), visible=False)
    # ax.legend()
    ax_cbar.set_yticks([])
    ax_cbar.set_ylabel("Prototype ID")
    ax_cbar.yaxis.set_label_position("right")
    sns.despine(ax=ax_cbar, bottom=True, left=True)


def plot_heatmap(distance, link_matrix, cluster_labels, prototypes, ax):
    """
    Plots a heatmap of the sorted distance matrix along with cluster and prototype
    information.

    Parameters
    ----------
    distance : numpy.ndarray
        Precomputed distance matrix.
    link_matrix : numpy.ndarray
        Linkage matrix obtained from hierarchical clustering.
    cluster_labels : numpy.ndarray
        Cluster labels for each sample.
    prototypes : numpy.ndarray
        Array of prototype indices corresponding to each cluster.
    ax : matplotlib.axes.Axes
        Axes object to draw the plot onto.

    Returns
    -------
    None

    """
    n_clusters = len(np.unique(cluster_labels))

    ax_cbar = ax.inset_axes([0.03, 0.13, 0.1, 0.02])

    # Sort distance matrix
    indices = hierarchy.leaves_list(link_matrix)
    sorted_distance = distance[indices, :][:, indices]

    # Plot heatmap
    im = ax.imshow(
        sorted_distance,
        cmap="Reds_r",
        aspect="auto",
    )

    # Add colorbar
    plt.colorbar(
        mappable=im,
        cax=ax_cbar,
        orientation="horizontal",
        label="distance",
    )

    # Highlight clusters and prototypes
    res, ind = np.unique(cluster_labels[indices], return_index=True)
    loc = 0
    for cid in res[np.argsort(ind)]:
        width = np.sum(cluster_labels == cid)
        ax.add_patch(
            Rectangle(
                (loc - 0.5, loc - 0.5),
                width,
                width,
                fill=False,
                edgecolor="k",
                lw=1,
            )
        )

        loc += width

    for cid in range(n_clusters):
        proto_loc = np.where(indices == prototypes[cid])[0]
        ax.scatter(
            proto_loc,
            proto_loc,
            marker="o",
            s=15,
            fc="white",
            ec="blue",
            label="HMM prototype" if cid == 1 else None,
        )

    # Finalize plot
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_yticks(range(len(indices)), labels=indices)
    ax.set_xlabel("Motif ID")
    ax.set_ylabel("Motif ID")
    ax.yaxis.set_label_position("right")
    ax.legend()
    sns.despine(ax=ax, bottom=True, left=True)


def plot_dendrogram(linkage_matrix, cluster_labels, ax):
    """
    Plots a dendrogram for hierarchical clustering with clusters colored according to
    labels.

    Parameters
    ----------
    linkage_matrix : numpy.ndarray
        Linkage matrix obtained from hierarchical clustering.
    cluster_labels : numpy.ndarray
        Cluster labels for each sample.
    ax : matplotlib.axes.Axes
        Axes object to draw the plot onto.

    Returns
    -------
    None

    """
    # NOTE: Coloring a dendrogram is a nightmare! For this reason we use a workaround.
    #       In brief, we estimate the threshold to get the right number of clusters and
    #       let the dendrogram function cycle over the colors as usual.
    n_clusters = len(np.unique(cluster_labels))
    indices = hierarchy.leaves_list(linkage_matrix)

    # Configure colormap
    cmap = get_custom_cmap(n_clusters)
    res, ind = np.unique(cluster_labels[indices], return_index=True)
    hierarchy.set_link_color_palette(
        [mpl.colors.rgb2hex(cmap.colors[i][:3]) for i in res[np.argsort(ind)]]
    )

    # Find color threshold
    def thr_minf(thr):
        # Count the number of "above" U links (easier), must be exactly n_clusters - 1
        return np.abs(np.sum(linkage_matrix[:, 2] > thr) - (n_clusters - 1))

    res = direct(
        thr_minf,
        bounds=[(min(linkage_matrix[:, 2]), max(linkage_matrix[:, 2]))],
        f_min=0,
        f_min_rtol=0,
    )

    color_thr = res.x[0]

    hierarchy.dendrogram(
        linkage_matrix,
        orientation="left",
        no_labels=True,
        color_threshold=color_thr,
        above_threshold_color="k",
        ax=ax,
    )

    # Finalize plot
    ax.set_ylabel("Motif hierarchy")
    ax.invert_yaxis()
    ax.set_xticks([])
    sns.despine(ax=ax, bottom=True, left=True)
