# pylint: disable=possibly-used-before-assignment

# Most imports are made after the parameters printing section because Python imports are just so fucking slow
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import attrs
import colorlog
from rich.traceback import install

install(show_locals=True, width=200)


# Logging
def get_logger(log_file_path: Path) -> logging.Logger:
    term_handler = logging.StreamHandler(sys.stdout)
    term_handler.setFormatter(
        colorlog.ColoredFormatter(
            "[%(cyan)s%(asctime)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )
    )
    term_handler.setLevel(logging.INFO)

    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
    file_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(term_handler)
    logger.addHandler(file_handler)
    return logger


#######################################################################################################################
######################################################## Utils ########################################################
#######################################################################################################################
def adapt_dataset_get_dataloader(dataset: DataSet, batch_size: int):
    # Checks
    assert dataset.dataset_params is not None, "Dataset parameters must be defined in the dataset instance"

    logger.info(f"Base dataset: {dataset}")

    # Data transforms
    # Globally speaking, we would like to predict the time using the *same transformations* than those used in the data loading of the generative model.
    # The (random) augmentations however must be discarded!
    #
    # *But* we also would like to use DINO's preprocessor...
    transforms_to_remove = [RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotationSquareSymmetry]
    logger.debug(f"\nRemoving transforms:{['.'.join([c.__module__, c.__name__]) for c in transforms_to_remove]}")

    used_transforms = Compose(
        [t for t in dataset.transforms.transforms if not any(isinstance(t, tr) for tr in transforms_to_remove)]
    )
    dataset.transforms = used_transforms
    logger.warning(f"\nNow using transforms: {used_transforms}")

    all_samples = list(Path(dataset.path).rglob(f"*.{dataset.dataset_params.file_extension}", recurse_symlinks=True))
    ds = ImageDataset(all_samples, dataset.transforms, dataset.expected_initial_data_range)  # pyright: ignore[reportAssignmentType]
    logger.info(f"Instantiated dataset: {ds}")

    # Custom data loading:
    # Needed to keep the connection between data and data path

    # Monkeypatch the dataset instance to return tensors+paths instead of tensors only
    def getitem_with_name(self, index: int) -> tuple[Tensor, Path]:
        path = self.samples[index]
        sample = self.load_to_pt_and_transform(path)
        return sample, path

    def getitems_with_names(self, indexes: list[int]) -> tuple[list[Tensor], list[Path]]:
        paths = [self.samples[idx] for idx in indexes]
        samples = self.get_items_by_name(paths)
        return samples, paths

    def collate_with_paths(batch):
        """Custom collate function that returns tensors and paths separately"""
        assert isinstance(batch, tuple), f"Expected batch to be a tuple, got {type(batch)}"
        assert len(batch) == 2, f"Expected batch to have 2 elements, got {len(batch)}"
        assert isinstance(batch[0], list), f"Expected first element of batch to be a list, got {type(batch[0])}"
        assert isinstance(batch[1], list), f"Expected second element of batch to be a list, got {type(batch[1])}"
        tensors, paths = batch[0], batch[1]
        assert len(tensors) == len(paths), (
            f"Expected tensors and paths to have the same length, got {len(tensors)} and {len(paths)}"
        )
        return {"tensors": torch.stack(tensors), "paths": paths}

    ds: torch.utils.data.Dataset[dict[str, Tensor | Path]]
    ds.__getitem__ = getitem_with_name.__get__(ds, type(ds))  # pylint: disable=no-value-for-parameter
    ds.__getitems__ = getitems_with_names.__get__(ds, type(ds))  # pylint: disable=no-value-for-parameter # pyright: ignore[reportAttributeAccessIssue]

    # Dataloader
    dl: DataLoader[dict[str, Tensor | Path]] = DataLoader(
        ds,
        batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_with_paths,
        prefetch_factor=1,
        persistent_workers=True,
    )

    return dl


def save_encodings(
    device: str,
    model_name: str,
    dataset: DataSet,
    use_model_preprocessor: bool,
    base_save_path: Path,
    save_name: str,
    batch_size: int,
    recompute_encodings: Literal[True] | Literal["no-overwrite"],
):
    # Checks
    assert dataset.dataset_params is not None, "Dataset parameters must be defined in the dataset instance"

    if (base_save_path / save_name).exists():
        if recompute_encodings == "no-overwrite":
            logger.info(
                f"Encodings already exist at {base_save_path / save_name}. Skipping recomputation since recompute_encodings is set to 'no-overwrite'"
            )
            return
        else:
            (base_save_path / save_name).unlink()
    base_save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving new encodings to {base_save_path / save_name}")
    # Dataloader
    dataloader = adapt_dataset_get_dataloader(dataset, batch_size)

    # Model
    logger.info(f"Loading model {model_name} on device {device}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Dinov2WithRegistersModel.from_pretrained(model_name).to(device)  # pyright: ignore[reportArgumentType]

    # Run
    rows: list[dict] = []

    for batch in tqdm(iter(dataloader), total=len(dataloader), desc="Processing batches"):
        # `tensors` is only used if use_DINO_preprocessor is `False`!
        tensors, paths = batch["tensors"], batch["paths"]

        # images
        if use_model_preprocessor:
            images = [Image.open(p) for p in paths]
            inputs = processor(images=images, return_tensors="pt").to(device)
        else:
            inputs = {"pixel_values": tensors.to(device)}

        # forward pass
        outputs = model(**inputs)

        # get the [CLS] token
        cls_tokens = outputs.pooler_output.cpu().numpy()
        # check because it's never *really* clear where is the [CLS] token in the output....
        assert (cls_tokens == outputs.last_hidden_state.cpu().numpy()[:, 0, :]).all()

        # labels
        labels: list[int] | list[str] = [dataset.dataset_params.key_transform(p.parent.name) for p in paths]

        # save each row of data to rows list
        for this_cls_token, this_label, this_path in zip(cls_tokens, labels, paths, strict=True):
            row_data = {
                "encodings": this_cls_token,
                "labels": this_label,
                "file_paths": str(this_path),  # Optional: save file paths for reference
            }
            rows.append(row_data)

    # Convert to DataFrame and save
    df = pd.DataFrame(rows)
    df.to_parquet(base_save_path / save_name)
    logger.info(f"Saved data to {base_save_path / save_name}")

    # smol print
    logger.info("DataFrame head:")
    logger.info(df.head())

    # return DataFrame
    return df


def plot_2D_embeddings(
    unique_labels: list[str],
    labels: np.ndarray,
    projector: PCA | LinearDiscriminantAnalysis | umap.UMAP,
    projector_embeddings: np.ndarray,
    encoding_scheme: str,
    dataset: DataSet,
    rng: Generator,
    viz_name: str,
    base_save_path: Path,
    subtitle: str | None = None,
    xy_labels: tuple[str, str] | None = None,
    spline_values: np.ndarray | None = None,
    projection_pairs: tuple[np.ndarray, np.ndarray] | None = None,
    centroids_in_base_embedding_space: np.ndarray | None = None,
    projection_type: Literal["base embedding space", "LDA embedding space"] = "base embedding space",
):
    plt.figure(figsize=(10, 10))
    # Use a sequential color palette with as many colors as unique labels
    palette = sns.color_palette("viridis", n_colors=len(unique_labels))
    label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

    # Make the plot order random
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    shuffled_embeddings = projector_embeddings[indices]
    shuffled_labels = [labels[i] for i in indices]
    shuffled_colors = [label_to_color[str(label)] for label in shuffled_labels]

    plt.scatter(shuffled_embeddings[:, 0], shuffled_embeddings[:, 1], c=shuffled_colors, s=10, alpha=0.5)
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()

    plt.gca().set_aspect("equal", "datalim")
    plt.suptitle(f"{viz_name.split('_')[0]} projection of CLS tokens of {encoding_scheme} on {dataset.name} dataset")
    if subtitle is not None:
        plt.title(subtitle)
    if xy_labels is not None:
        plt.xlabel(xy_labels[0])
        plt.ylabel(xy_labels[1])
    else:
        plt.xlabel(f"{viz_name} 1")
        plt.ylabel(f"{viz_name} 2")

    # 2. Plot spline values
    if spline_values is not None:
        if projection_type == "base embedding space":
            spline_projector_emb: np.ndarray = projector.transform(spline_values)  # pyright: ignore[reportAssignmentType]
            plt.plot(
                spline_projector_emb[:, 0],
                spline_projector_emb[:, 1],
                color="red",
                linewidth=1.5,
                label="centroid-spline",
            )
        elif projection_type == "LDA embedding space" and isinstance(projector, LinearDiscriminantAnalysis):
            plt.plot(
                spline_values[:, 0],
                spline_values[:, 1],
                color="red",
                linewidth=1.5,
                label="centroid-spline",
            )
        else:
            logger.debug(
                f"Not plotting spline values for {projection_type} projection type with {type(projector)} projector"
            )

    # 3. Plot some projections on spline
    if projection_pairs is not None:
        assert len(projection_pairs) == 2, (
            f"Expected projection_pairs to have shape (2, n_points, embedding_dim), got {projection_pairs.shape}"
        )
        source_coord, projected_coord = projection_pairs[0], projection_pairs[1]
        assert source_coord.shape == projected_coord.shape, (
            f"Expected source and projected coordinates to have the same shape, got {source_coord.shape} and {projected_coord.shape}"
        )
        if projection_type == "base embedding space":
            source_coord_projector_emb: np.ndarray = projector.transform(source_coord)  # pyright: ignore[reportAssignmentType]
            projected_time_coord_projector_emb: np.ndarray = projector.transform(projected_coord)  # pyright: ignore[reportAssignmentType]
            for idx in range(len(source_coord)):
                plt.plot(
                    [source_coord_projector_emb[idx, 0], projected_time_coord_projector_emb[idx, 0]],
                    [source_coord_projector_emb[idx, 1], projected_time_coord_projector_emb[idx, 1]],
                    color="gray",
                    linewidth=1,
                    label="Projection Line" if idx == 0 else None,
                )
        elif projection_type == "LDA embedding space" and isinstance(projector, LinearDiscriminantAnalysis):
            for idx in range(len(source_coord)):
                plt.plot(
                    [source_coord[idx, 0], projected_coord[idx, 0]],
                    [source_coord[idx, 1], projected_coord[idx, 1]],
                    color="gray",
                    linewidth=1,
                    label="Projection Line" if idx == 0 else None,
                )
        else:
            logger.debug(
                f"Not plotting projection pairs for {projection_type} projection type with {type(projector)} projector"
            )

    # 4. Add time centroids
    if centroids_in_base_embedding_space is not None:
        centroids_projector_embeddings: np.ndarray = projector.transform(centroids_in_base_embedding_space)  # pyright: ignore[reportAssignmentType]
        plt.scatter(
            centroids_projector_embeddings[:, 0],
            centroids_projector_embeddings[:, 1],
            marker="*",
            s=30,
            c="gold",
            edgecolor="black",
            label="Centroids",
        )

    # 5. Add legend and restore original axes limits
    class_handles = [
        mlines.Line2D([], [], color=label_to_color[label], marker="o", linestyle="None", markersize=8, label=str(label))
        for label in unique_labels
    ]
    # Add handles for spline, projections, centroids if present
    extra_handles = []
    if spline_values is not None:
        extra_handles.append(mlines.Line2D([], [], color="red", linestyle="-", linewidth=2, label="Centroid spline"))
    if projection_pairs is not None:
        extra_handles.append(mlines.Line2D([], [], color="gray", linestyle="-", linewidth=1.5, label="Projection Line"))
    if centroids_in_base_embedding_space is not None:
        extra_handles.append(
            mlines.Line2D(
                [],
                [],
                color="gold",
                marker="*",
                linestyle="None",
                markersize=10,
                markeredgecolor="black",
                label="Centroids",
            )
        )
    plt.legend(handles=class_handles + extra_handles, title="Legend")
    plt.xlim(xlim)
    plt.ylim(ylim)

    save_path = base_save_path / f"2d_{viz_name.lower()}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    logger.debug(f"Saved 2D plot to {save_path}")


def plot_histograms_embeddings_means_vars(
    tokens: np.ndarray, encoding_scheme: str, dataset_name: str, base_save_path: Path
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Top: means
    sns.histplot(x=tokens.mean(axis=0), bins=100, ax=ax1, color="skyblue")
    ax1.set_title("Histogram of token means")
    ax1.set_xlabel("Mean value")
    ax1.set_ylabel("Count")

    # Bottom: variances
    sns.histplot(x=tokens.var(axis=0), bins=100, ax=ax2, color="salmon")
    ax2.set_title("Histogram of token variances")
    ax2.set_xlabel("Variance value")
    ax2.set_ylabel("Count")

    plt.suptitle(f"Histograms of means and variances of CLS token of {encoding_scheme} on {dataset_name}", fontsize=14)
    plt.tight_layout()
    save_path = base_save_path / "CLS_tokens_stats_histogram.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.debug(f"Saved histogram of means and vars of {encoding_scheme} CLS tokens to {save_path}")


def plot_histograms_continuous_time_preds(
    base_save_path: Path,
    continuous_time_predictions: np.ndarray,
    labels: np.ndarray,
    basename: str,
    encoding_scheme: str,
    dataset_name: str,
    y_log_scale: bool = False,
):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 18))

    labels_type = type(labels[0])
    hue_order = [labels_type(label) for label in sorted_unique_labels]

    # Top: stacked histogram
    sns.histplot(
        x=continuous_time_predictions,
        hue=labels,
        bins=100,
        palette="viridis",
        multiple="stack",
        ax=ax1,
        hue_order=hue_order,
    )
    ax1.set_title("Stacked histogram of continuous time predictions")
    ax1.set_xlabel("Continuous time prediction")
    if y_log_scale:
        ax1.set_yscale("log")
        ax1.set_ylabel("Raw count (log scale)")
    else:
        ax1.set_ylabel("Raw count")

    # Middle: layered histogram per class
    sns.histplot(
        x=continuous_time_predictions,
        hue=labels,
        bins=100,
        palette="viridis",
        element="step",
        ax=ax2,
        stat="percent",
        common_norm=False,  # each class sums to 100%
        alpha=0.5,
        hue_order=hue_order,
    )
    ax2.set_title("Layered histogram of continuous time predictions - per-class normalized")
    ax2.set_xlabel("Continuous time prediction")
    ax2.set_ylabel("Percentage of samples (per class)")
    ax2.get_legend().set_title("True label")

    # Bottom: per-class normalized histogram
    sns.histplot(
        x=continuous_time_predictions,
        hue=labels,
        bins=100,
        palette="viridis",
        ax=ax3,
        multiple="fill",
        hue_order=hue_order,
    )
    ax3.set_title("Filled histogram")
    ax3.set_xlabel("Continuous time prediction")
    ax3.set_ylabel("Percentage of samples (per time bin)")
    ax3.get_legend().set_title("True label")

    plt.suptitle(
        f"Continuous time predictions histogram for {basename} with {encoding_scheme} on {dataset_name}", fontsize=14
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_path = base_save_path / f"continuous_time_predictions_{basename}_histogram.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.debug(f"Saved histogram of continuous time predictions to {save_path}")


def plot_boxplots_continuous_time_preds(
    base_save_path: Path,
    labels: np.ndarray,
    sorted_unique_labels: list[str],
    continuous_time_predictions: np.ndarray,
    basename: str,
    plot_swarmplot: bool = False,
):
    if plot_swarmplot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    else:
        fig, ax1 = plt.subplots(figsize=(14, 6))

    labels_type = type(labels[0])
    hue_order = [labels_type(label) for label in sorted_unique_labels]

    # Subplot 1: Boxplot and Stripplot
    sns.boxplot(
        x=labels,
        y=continuous_time_predictions,
        palette="tab10",
        hue=labels,
        order=hue_order,
        boxprops={"alpha": 0.5},
        showfliers=False,
        ax=ax1,
        legend=False,
    )
    sns.stripplot(
        x=labels,
        y=continuous_time_predictions,
        size=2.5,
        alpha=0.7,
        palette="tab10",
        hue=labels,
        order=hue_order,
        jitter=0.3,
        ax=ax1,
        legend=False,
    )
    sns.despine(ax=ax1)
    ax1.set_xlabel("True labels")
    ax1.set_ylabel("Continuous time predictions")
    ax1.set_title("Boxplot with Stripplot")

    # Subplot 2: Boxplot and maybe Swarmplot
    if plot_swarmplot:
        sns.boxplot(
            x=labels,
            y=continuous_time_predictions,
            palette="tab10",
            hue=labels,
            order=hue_order,
            boxprops={"alpha": 0.5},
            showfliers=False,
            ax=ax2,  # pyright: ignore[reportPossiblyUnboundVariable]
            legend=False,
        )
        sns.swarmplot(
            x=labels,
            y=continuous_time_predictions,
            size=2,
            alpha=0.7,
            palette="tab10",
            hue=labels,
            order=hue_order,
            ax=ax2,  # pyright: ignore[reportPossiblyUnboundVariable]
            legend=False,
        )
        sns.despine(ax=ax2)  # pyright: ignore[reportPossiblyUnboundVariable]
        ax2.set_xlabel("True labels")  # pyright: ignore[reportPossiblyUnboundVariable]
        ax2.set_ylabel("Continuous time predictions")  # pyright: ignore[reportPossiblyUnboundVariable]
        ax2.set_title("Boxplot with Swarmplot")  # pyright: ignore[reportPossiblyUnboundVariable]

    # Add a single legend for the figure if desired, or rely on swarmplot's legend
    handles, legend_labels = ax1.get_legend_handles_labels()
    fig.legend(handles, legend_labels, title="Labels", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.suptitle(f"True labels vs continuous time predictions for {basename}", fontsize=16)

    save_path = base_save_path / f"continuous_time_predictions_{basename}_boxplots.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.debug(f"Saved boxplots of continuous time predictions to {save_path}")


# def plot_ellipse(mean: np.ndarray, cov: np.ndarray, color: str, ax: plt.Axes) -> None:
#     # draw a 2-sigma ellipse from a mean & covariance
#     v, w = np.linalg.eigh(cov)
#     u = w[0] / np.linalg.norm(w[0])
#     angle = np.degrees(np.arctan2(u[1], u[0]))
#     ell = mpl.patches.Ellipse(
#         xy=mean,
#         width=2 * np.sqrt(v[0]),
#         height=2 * np.sqrt(v[1]),
#         angle=180 + angle,
#         facecolor=color,
#         edgecolor="black",
#         linewidth=2,
#         alpha=0.4,
#     )
#     ell.set_clip_box(ax.bbox)
#     ax.add_artist(ell)


# def plot_result(estimator: LinearDiscriminantAnalysis, X: np.ndarray, y: np.ndarray, ax: plt.Axes) -> None:
#     # prepare a viridis palette for N classes
#     classes = estimator.classes_
#     palette = sns.color_palette("viridis", n_colors=len(classes))
#     cmap = colors.ListedColormap(palette)

#     # decision‐region background
#     DecisionBoundaryDisplay.from_estimator(
#         estimator,
#         X,
#         response_method="predict",
#         plot_method="pcolormesh",
#         ax=ax,
#         cmap=cmap,
#         alpha=0.3,
#     )

#     # scatter the points
#     scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=40)
#     ax.legend(*scatter.legend_elements(), title="Class")

#     # class means
#     ax.scatter(  # this is incorrect: we should project the means to the 2D space
#         estimator.means_[:, 0],
#         estimator.means_[:, 1],
#         c="yellow",
#         s=200,
#         marker="*",
#         edgecolor="black",
#     )

#     # plot ellipses (LDA uses a shared covariance)
#     covs = [estimator.covariance_] * len(classes)
#     for mean, cov, col in zip(estimator.means_, covs, palette):
#         plot_ellipse(mean, cov, col, ax)

#     ax.set_box_aspect(1)
#     for sp in ax.spines.values():
#         sp.set_visible(False)
#     ax.set(xticks=[], yticks=[])


def plot_3D_embeddings(
    labels: np.ndarray,
    sorted_unique_labels: list[str],
    projector_embeddings: np.ndarray,
    projector: PCA | LinearDiscriminantAnalysis | umap.UMAP,
    rng: Generator,
    base_save_path: Path,
    dataset: DataSet,
    viz_name: str,
    encoding_scheme: str,
    subtitle: str | None = None,
    xyz_labels: tuple[str, str, str] | None = None,
    spline_values: np.ndarray | None = None,
    projection_pairs: tuple[np.ndarray, np.ndarray] | None = None,
    centroids_in_base_embedding_space: np.ndarray | None = None,
    projection_type: Literal["base embedding space", "LDA embedding space"] = "base embedding space",
):
    """
    Plot 3D embeddings of some projector (e.g. PCA, LDA) with time labels.
    Additionally plot class centroids if projector is LDA.
    Additionally plot a fitted spline and the projections of some given points onto that spline.
    """
    # 1. Plot embeddings in 3D projector space
    # Use a sequential color palette with as many colors as unique labels
    palette = sns.color_palette("viridis", n_colors=len(sorted_unique_labels))
    color_discrete_map = {
        str(label): f"rgb{tuple(int(255 * c) for c in palette[i])}" for i, label in enumerate(sorted_unique_labels)
    }
    # Make the plot order random
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    shuffled_projector_embeddings = projector_embeddings[indices]
    shuffled_labels = [labels[i] for i in indices]

    if xyz_labels is None:
        xyz_labels = (f"{viz_name} 1", f"{viz_name} 2", f"{viz_name} 3")

    title = f"3D {viz_name.split('_')[0]} projection of CLS tokens of {encoding_scheme} from {dataset.name}"
    if subtitle is not None:
        title += f"<br><sub>{subtitle}</sub>"

    fig = px.scatter_3d(
        x=shuffled_projector_embeddings[:, 0],
        y=shuffled_projector_embeddings[:, 1],
        z=shuffled_projector_embeddings[:, 2],
        color=[str(label) for label in shuffled_labels],
        color_discrete_map=color_discrete_map,
        category_orders={"color": [str(label) for label in sorted_unique_labels]},
        labels={"x": xyz_labels[0], "y": xyz_labels[1], "z": xyz_labels[2], "color": "Label"},
        title=title,
        opacity=0.5,
    )
    fig.update_traces(marker=dict(size=2))
    # Enforce 1:1:1 aspect ratio
    fig.update_scenes(aspectmode="cube")
    # Set default view axes limits neglecting the spline
    x_min, x_max = shuffled_projector_embeddings[:, 0].min(), shuffled_projector_embeddings[:, 0].max()
    y_min, y_max = shuffled_projector_embeddings[:, 1].min(), shuffled_projector_embeddings[:, 1].max()
    z_min, z_max = shuffled_projector_embeddings[:, 2].min(), shuffled_projector_embeddings[:, 2].max()
    fig.update_scenes(
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        zaxis=dict(range=[z_min, z_max]),
    )

    # 2. Add time centroids
    if centroids_in_base_embedding_space is not None:
        centroids_projector_embeddings: np.ndarray = projector.transform(centroids_in_base_embedding_space)  # pyright: ignore[reportAssignmentType]
        fig.add_trace(
            go.Scatter3d(
                x=centroids_projector_embeddings[:, 0],
                y=centroids_projector_embeddings[:, 1],
                z=centroids_projector_embeddings[:, 2],
                mode="markers",
                marker=dict(symbol="cross", size=7, color="darkgoldenrod"),
                name="centroids",
            )
        )

    # 3. Plot spline values
    if spline_values is not None:
        if projection_type == "base embedding space":
            spline_projector_emb: np.ndarray = projector.transform(spline_values)  # pyright: ignore[reportAssignmentType]
            fig.add_trace(
                go.Scatter3d(
                    x=spline_projector_emb[:, 0],
                    y=spline_projector_emb[:, 1],
                    z=spline_projector_emb[:, 2],
                    mode="lines",
                    line=dict(color="red", width=3),
                    name="centroid-spline",
                )
            )
        elif projection_type == "LDA embedding space" and isinstance(projector, LinearDiscriminantAnalysis):
            fig.add_trace(
                go.Scatter3d(
                    x=spline_values[:, 0],
                    y=spline_values[:, 1],
                    z=spline_values[:, 2],
                    mode="lines",
                    line=dict(color="red", width=3),
                    name="centroid-spline",
                )
            )
        else:
            logger.debug(
                f"Not plotting spline values for {projection_type} projection type with {type(projector)} projector"
            )

    # 4. Plot some projections on spline
    if projection_pairs is not None:
        assert len(projection_pairs) == 2, (
            f"Expected projection_pairs to have shape (2, n_points, embedding_dim), got {projection_pairs.shape}"
        )
        source_coord, projected_coord = projection_pairs[0], projection_pairs[1]
        assert source_coord.shape == projected_coord.shape, (
            f"Expected source and projected coordinates to have the same shape, got {source_coord.shape} and {projected_coord.shape}"
        )
        if projection_type == "base embedding space":
            source_coord_projector_emb: np.ndarray = projector.transform(source_coord)  # pyright: ignore[reportAssignmentType]
            projected_time_coord_projector_emb: np.ndarray = projector.transform(projected_coord)  # pyright: ignore[reportAssignmentType]
            for idx in range(len(source_coord)):
                fig.add_trace(
                    go.Scatter3d(
                        x=[source_coord_projector_emb[idx, 0], projected_time_coord_projector_emb[idx, 0]],
                        y=[source_coord_projector_emb[idx, 1], projected_time_coord_projector_emb[idx, 1]],
                        z=[source_coord_projector_emb[idx, 2], projected_time_coord_projector_emb[idx, 2]],
                        mode="lines",
                        line=dict(color="gray", width=2),
                        legendgroup="Projection Lines",
                        name="Projection Line" if idx == 0 else None,
                        showlegend=(idx == 0),
                    )
                )
        elif projection_type == "LDA embedding space" and isinstance(projector, LinearDiscriminantAnalysis):
            for idx in range(len(source_coord)):
                fig.add_trace(
                    go.Scatter3d(
                        x=[source_coord[idx, 0], projected_coord[idx, 0]],
                        y=[source_coord[idx, 1], projected_coord[idx, 1]],
                        z=[source_coord[idx, 2], projected_coord[idx, 2]],
                        mode="lines",
                        line=dict(color="gray", width=2),
                        legendgroup="Projection Lines",
                        name="Projection Line" if idx == 0 else None,
                        showlegend=(idx == 0),
                    )
                )
        else:
            logger.debug(
                f"Not plotting projection pairs for {projection_type} projection type with {type(projector)} projector"
            )

    # 5. Save and return
    save_path = base_save_path / f"3d_{viz_name}.html"
    fig.write_html(save_path, auto_open=False)
    logger.debug(f"Saved interactive 3D plot to {save_path}")

    png_save_path = save_path.with_suffix(".png")
    fig.update_layout(width=2100, height=1500)
    fig.write_image(png_save_path, scale=2)
    logger.debug(f"Saved 3D plot to {png_save_path}")


def project_to_time(points: np.ndarray, spline: BSpline, t_min: float, t_max: float) -> np.ndarray:
    """
    ## Arguments:
    - `points`: `ndarray` of shape (n_samples, n_features)
    - `spline`: `BSpline`
    - `t_min`: minimum time value
    - `t_max`: maximum time value

    ## Returns:
    - `times`: `ndarray`, continuous times array of shape (n_samples,)
    """
    assert spline(0).shape == points[0].shape, (
        f"Expected spline to output the same shape as points, got {spline(0).shape} and {points[0].shape}"
    )

    def find_t(pt: np.ndarray) -> float:
        obj = lambda t: np.sum((spline(t) - pt) ** 2)
        res: OptimizeResult = minimize_scalar(obj, bounds=(t_min, t_max), method="bounded")  # pyright: ignore[reportAssignmentType]
        assert res.success, f"Optimization failed for point {pt} with message: {res.message}"
        return res.x

    times = np.array([find_t(p) for p in points])

    return times


def plot_histograms_distances_to_true_labels(
    sorted_unique_label_times: list[float],
    sorted_unique_labels: list[str],
    continuous_times: np.ndarray,
    dataset_name: str,
    base_save_path: Path,
    labels: np.ndarray,
    viz_name: str,
):
    dists_to_true_label: list[float] = []
    nb_changes = 0
    changes_count_from_true_label = dict.fromkeys(sorted_unique_labels, 0)
    possible_diffs = sorted(
        set(abs(a - b) for i, a in enumerate(sorted_unique_label_times) for b in sorted_unique_label_times[i:])
    )
    changes_count_by_dist = dict.fromkeys(possible_diffs, 0)

    for true_label, time in zip(labels, continuous_times, strict=True):
        time: float
        true_label_time = sorted_unique_label_times[sorted_unique_labels.index(str(true_label))]
        dist_to_true_label = abs(true_label_time - time)
        dists_to_true_label.append(dist_to_true_label)

        closest_label_time = min(sorted_unique_label_times, key=lambda x: abs(x - time))
        closest_label = sorted_unique_labels[sorted_unique_label_times.index(closest_label_time)]

        if closest_label != true_label:
            nb_changes += 1
            changes_count_from_true_label[str(true_label)] += 1

        changes_count_by_dist[abs(true_label_time - closest_label_time)] += 1

    logger.debug(f"Number of changes: {nb_changes} out of {len(labels)} ({nb_changes / len(labels) * 100:.2f}%)")
    logger.debug(f"Changes count from true label: {changes_count_from_true_label}")
    logger.debug(
        f"Distances distribution between true label and true label closest to time prediction: {changes_count_by_dist}"
    )

    n_labels = len(sorted_unique_labels)
    n_cols = min(4, n_labels)
    n_rows = (n_labels + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(4 * n_cols, 4 * (n_rows + 1)))
    gs = gridspec.GridSpec(nrows=n_rows + 1, ncols=n_cols, height_ratios=[1.2] + [1] * n_rows)

    # Top: combined histogram spanning all columns
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.hist(dists_to_true_label, bins=256)
    ax_top.set_title("Histogram of distances to true label")
    ax_top.set_xlabel("Distance to true label")
    ax_top.set_ylabel("Count")
    ax_top.set_xlim(0, 1)

    # Grid of per-label histograms
    for idx, label in enumerate(sorted_unique_labels):
        row = idx // n_cols + 1
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        label_mask = np.array([str(lab) == str(label) for lab in labels])
        ax.hist(np.array(dists_to_true_label)[label_mask], bins=100, color="tab:blue")
        ax.set_title(label)
        ax.set_xlabel("Distance to true label")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)

    plt.suptitle(f"Distances to true label histograms for {dataset_name}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_path = base_save_path / f"continuous_time_predictions_distances_to_true_labels_{viz_name}_histogram.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.debug(f"Saved histogram of distances to true labels to {save_path}")


def fit_spline_project_time_plots(
    projection_type: Literal["base embedding space", "LDA embedding space"],
    sorted_centroids_in_base_embedding_space: np.ndarray,
    sorted_unique_labels: list[str],
    sorted_unique_label_times: list[float],
    params: Params,
    cls_tokens: np.ndarray,
    labels: np.ndarray,
    ds: DataSet,
    rng: Generator,
    umap_2d_reducer: umap.UMAP,
    umap_2d_embeddings: np.ndarray,
    umap_3d_reducer: umap.UMAP,
    umap_3d_embeddings: np.ndarray,
    pca: PCA,
    pca_embeddings: np.ndarray,
    lda: LinearDiscriminantAnalysis,
    lda_embeddings: np.ndarray,
    this_run_save_path_this_ds: Path,
    encoding_scheme: str,
    pca_explained_variance: np.ndarray,
    lda_explained_variance: np.ndarray,
    df: pd.DataFrame,
    normalization_method: str,
) -> np.ndarray:
    if projection_type == "base embedding space":
        # Compute spline
        logger.debug(
            f"Fitting interpolating spline through class centroids of base embedding space of {encoding_scheme}"
        )
        logger.debug(
            f"Centroids shape: {sorted_centroids_in_base_embedding_space.shape} | true labels: {sorted_unique_labels} | true label times {sorted_unique_label_times}"
        )
        spline = make_interp_spline(sorted_unique_label_times, sorted_centroids_in_base_embedding_space)
        # Evaluate spline
        assert (sorted_unique_label_times[0], sorted_unique_label_times[-1]) == (0, 1)
        t_min, t_max = -params.frac_range_delta, 1 + params.frac_range_delta
        logger.warning(
            f"Using {params.frac_range_delta} of the 0-1 time range for t_min, t_max = {t_min}, {t_max} parametrization of the spline"
        )
        times_to_eval_spline = np.linspace(t_min, t_max, 1000)
        spline_values = spline(times_to_eval_spline)  # these are in DINO's embedding space
        # Project base embeddings on spline
        continuous_time_predictions = project_to_time(cls_tokens, spline, t_min, t_max)
        logger.debug(
            f"Computed continuous time predictions from spline projection, shape: {continuous_time_predictions.shape}, excerpt: {continuous_time_predictions[:5]}"
        )
        # Plot spline and projections along with base embeddings
        random_idx_to_plot_projections = rng.choice(len(cls_tokens), size=50, replace=False)
        projection_pairs = (
            cls_tokens[random_idx_to_plot_projections],
            spline(continuous_time_predictions[random_idx_to_plot_projections]),
        )
    elif projection_type == "LDA embedding space":
        # Compute spline
        logger.debug("Fitting interpolating spline through class centroids of LDA embedding space")
        centroids_in_lda_embedding_space = lda.transform(sorted_centroids_in_base_embedding_space)
        logger.debug(
            f"Centroids shape: {centroids_in_lda_embedding_space.shape} | true labels: {sorted_unique_labels} | true label times {sorted_unique_label_times}"
        )
        spline = make_interp_spline(sorted_unique_label_times, centroids_in_lda_embedding_space)
        # Evaluate spline
        assert (sorted_unique_label_times[0], sorted_unique_label_times[-1]) == (0, 1)
        t_min, t_max = -params.frac_range_delta, 1 + params.frac_range_delta
        logger.warning(
            f"Using {params.frac_range_delta} of the 0-1 time range for t_min, t_max = {t_min}, {t_max} parametrization of the spline"
        )
        times_to_eval_spline = np.linspace(t_min, t_max, 1000)
        spline_values = spline(times_to_eval_spline)  # these are in LDA's embedding space
        # Project lda embeddings on spline
        continuous_time_predictions = project_to_time(lda_embeddings, spline, t_min, t_max)
        logger.debug(
            f"Computed continuous time predictions from spline projection, shape: {continuous_time_predictions.shape}, excerpt: {continuous_time_predictions[:5]}"
        )
        # Plot spline and projections along with base embeddings
        random_idx_to_plot_projections = rng.choice(len(cls_tokens), size=50, replace=False)
        projection_pairs = (
            lda.transform(cls_tokens[random_idx_to_plot_projections]),
            spline(continuous_time_predictions[random_idx_to_plot_projections]),
        )
    else:
        raise ValueError(
            f"Unknown projection type: {projection_type}. Expected 'base embedding space' or 'LDA embedding space'."
        )

    vizes_name_suffix = f"spline_projection_on_{projection_type.replace(' ', '_')}"

    # plot 2D & 3D embeddings with spline and projections for LDA, UMAP and PCA viz
    if projection_type == "base embedding space":
        plot_2D_embeddings(
            sorted_unique_labels,
            labels,
            umap_2d_reducer,
            umap_2d_embeddings,
            params.model_name,
            ds,
            rng,
            f"UMAP_{vizes_name_suffix}",
            this_run_save_path_this_ds,
            f"Seed={params.seed} | params.frac_range_delta={params.frac_range_delta}",
            ("UMAP 1", "UMAP 2"),
            spline_values,
            projection_pairs,
            sorted_centroids_in_base_embedding_space,  # pyright: ignore[reportArgumentType]
            projection_type,
        )
        plot_2D_embeddings(
            sorted_unique_labels,
            labels,
            pca,
            pca_embeddings,
            params.model_name,
            ds,
            rng,
            f"PCA_{vizes_name_suffix}",
            this_run_save_path_this_ds,
            f"Total explained variance: {np.sum(pca_explained_variance[:2]) * 100:.1f}%",
            (
                f"PCA 1 ({pca_explained_variance[0] * 100:.1f}% of explained variance)",
                f"PCA 2 ({pca_explained_variance[1] * 100:.1f}% of explained variance)",
            ),
            spline_values,
            projection_pairs,
            sorted_centroids_in_base_embedding_space,  # pyright: ignore[reportArgumentType]
            projection_type,
        )
    plot_2D_embeddings(
        sorted_unique_labels,
        labels,
        lda,
        lda_embeddings,
        params.model_name,
        ds,
        rng,
        f"LDA_{vizes_name_suffix}",
        this_run_save_path_this_ds,
        f"Total explained variance: {np.sum(lda_explained_variance[:2]) * 100:.1f}%",
        (
            f"LDA 1 ({lda_explained_variance[0] * 100:.1f}% of explained variance)",
            f"LDA 2 ({lda_explained_variance[1] * 100:.1f}% of explained variance)",
        ),
        spline_values,
        projection_pairs,
        sorted_centroids_in_base_embedding_space,  # pyright: ignore[reportArgumentType]
        projection_type,
    )
    if projection_type == "base embedding space":
        plot_3D_embeddings(
            labels,
            sorted_unique_labels,
            umap_3d_embeddings,
            umap_3d_reducer,
            rng,
            this_run_save_path_this_ds,
            ds,
            f"UMAP_{vizes_name_suffix}",
            encoding_scheme,
            f"Seed={params.seed} | params.frac_range_delta={params.frac_range_delta}",
            ("UMAP 1", "UMAP 2", "UMAP 3"),
            spline_values,
            projection_pairs,
            sorted_centroids_in_base_embedding_space,  # pyright: ignore[reportArgumentType]
            projection_type,
        )
        plot_3D_embeddings(
            labels,
            sorted_unique_labels,
            pca_embeddings,
            pca,
            rng,
            this_run_save_path_this_ds,
            ds,
            f"PCA_{vizes_name_suffix}",
            encoding_scheme,
            f"Total explained variance: {np.sum(pca_explained_variance[:3]) * 100:.1f}% | seed={params.seed} | params.frac_range_delta={params.frac_range_delta}",
            (
                f"PCA 1 ({pca_explained_variance[0] * 100:.1f}% of explained variance)",
                f"PCA 2 ({pca_explained_variance[1] * 100:.1f}% of explained variance)",
                f"PCA 3 ({pca_explained_variance[2] * 100:.1f}% of explained variance)",
            ),
            spline_values,
            projection_pairs,
            sorted_centroids_in_base_embedding_space,  # pyright: ignore[reportArgumentType]
            projection_type,
        )
    plot_3D_embeddings(
        labels,
        sorted_unique_labels,
        lda_embeddings,
        lda,
        rng,
        this_run_save_path_this_ds,
        ds,
        f"LDA_{vizes_name_suffix}",
        encoding_scheme,
        f"Total explained variance: {np.sum(lda_explained_variance[:3]) * 100:.1f}% | params.frac_range_delta={params.frac_range_delta}",
        (
            f"LDA 1 ({lda_explained_variance[0] * 100:.1f}% of explained variance)",
            f"LDA 2 ({lda_explained_variance[1] * 100:.1f}% of explained variance)",
            f"LDA 3 ({lda_explained_variance[2] * 100:.1f}% of explained variance)",
        ),
        spline_values,
        projection_pairs,
        sorted_centroids_in_base_embedding_space,  # pyright: ignore[reportArgumentType]
        projection_type,
    )

    # Plot histograms and boxplots of continuous time predictions
    plot_histograms_continuous_time_preds(
        this_run_save_path_this_ds,
        continuous_time_predictions,
        labels,
        vizes_name_suffix,
        encoding_scheme,
        ds.name,
    )
    plot_boxplots_continuous_time_preds(
        this_run_save_path_this_ds,
        labels,
        sorted_unique_labels,
        continuous_time_predictions,
        vizes_name_suffix,
        len(df) <= 10_000,
    )

    # plot distances to true labels
    plot_histograms_distances_to_true_labels(
        sorted_unique_label_times,
        sorted_unique_labels,
        continuous_time_predictions,
        ds.name,
        this_run_save_path_this_ds,
        labels,
        vizes_name_suffix,
    )

    # normalize to [0,1]
    match normalization_method:
        case "min-max":
            continuous_time_predictions -= continuous_time_predictions.min()
            continuous_time_predictions /= continuous_time_predictions.max()
        case "5perc-95perc":
            perc_5 = np.percentile(continuous_time_predictions, 5)
            perc_95 = np.percentile(continuous_time_predictions, 95)
            continuous_time_predictions = np.clip(continuous_time_predictions, perc_5, perc_95)
            continuous_time_predictions -= perc_5
            continuous_time_predictions /= perc_95 - perc_5
        case _:
            raise ValueError(f"Unknown normalization method: {normalization_method}")

    plot_histograms_continuous_time_preds(
        this_run_save_path_this_ds,
        continuous_time_predictions,
        labels,
        f"{vizes_name_suffix}_{normalization_method}_norm",
        encoding_scheme,
        ds.name,
    )

    return continuous_time_predictions


@attrs.define
class Params:
    datasets: list[DataSet]
    device: str
    model_name: str
    batch_size: int
    use_model_preprocessor: bool
    recompute_encodings: bool | Literal["no-overwrite"]
    save_times: Literal["no-overwrite"] | Literal["overwrite"] | Literal["ask-before-overwrite"] | Literal["no"]
    seed: int
    frac_range_delta: float  # fraction of the true time range to use for t_min, t_max parametrization


if __name__ == "__main__":
    ###################################################################################################################
    ################################################### Parameters ####################################################
    ###################################################################################################################
    sys.path.append(".")
    # Attention: we *might or might not* use our datasets' pipeline as DINO has its own preprocessing pipeline.
    from my_conf.dataset.BBBC021_196_docetaxel_inference import (
        BBBC021_196_docetaxel_inference as bbbc021_ds,  # noqa: E402
    )
    from my_conf.dataset.biotine_png_128_inference import dataset as biotine_ds  # noqa: E402
    from my_conf.dataset.chromalive6h_3ch_png_inference import dataset as chromalive_ds  # noqa: E402
    from my_conf.dataset.diabetic_retinopathy_inference import (
        diabetic_retinopathy_inference as diabetic_retinopathy_ds,  # noqa: E402
    )
    from my_conf.dataset.Jurkat_inference import Jurkat_inference as jurkat_ds  # noqa: E402
    from my_conf.dataset.NASH_fibrosis_inference import dataset as NASH_fibrosis_ds  # noqa: E402
    from my_conf.dataset.NASH_steatosis_inference import dataset as NASH_steatosis_ds  # noqa: E402

    # fmt: off
    params = Params(
        datasets               = [NASH_steatosis_ds, diabetic_retinopathy_ds, biotine_ds, bbbc021_ds, NASH_fibrosis_ds, chromalive_ds, jurkat_ds],
        device                 = "cuda:0",
        model_name             = "facebook/dinov2-with-registers-giant",
        batch_size             = 1024,
        use_model_preprocessor = False,
        recompute_encodings    = "no-overwrite",
        save_times             = "overwrite",
        seed                   = random.randint(0, 2**32 - 1),
        frac_range_delta       = 0.3,
    )
    # fmt: on

    ###################################################################################################################
    ##################################################### Launch ######################################################
    ###################################################################################################################
    base_save_dir = Path("ordering_datasets")
    encoding_scheme = params.model_name.replace("/", "_") + (
        "_model_preproc" if params.use_model_preprocessor else "_dataset_preproc"
    )
    this_run_save_path = base_save_dir / encoding_scheme
    this_run_save_path.mkdir(parents=True, exist_ok=True)
    logger = get_logger(this_run_save_path / "logs.log")
    logger.debug("")
    logger.info("-" * 120)
    logger.info("Starting new run")
    logger.info("=> Parameters:")
    params_print = {
        "Datasets": [ds.name for ds in params.datasets],
        "Base save dir": base_save_dir,
        "Run save path": this_run_save_path,
        **{k: v for k, v in attrs.asdict(params).items() if k != "datasets"},
    }
    label_width = max(len(param) for param in params_print) + 1  # ":"
    for param, value in params_print.items():
        label_str = f"{param}:"
        dots = "_" * (label_width - len(label_str))
        logger.info(f"    {label_str}{dots} {value}")
    print("", flush=True)
    inpt = input("=> Continue? (y/[]) ")
    if inpt != "y":
        logger.info("Exiting...")
        sys.exit(0)

    # ruff: noqa: E402
    logger.info("Loading imports... ")
    # Imports
    from typing import Literal

    import matplotlib.gridspec as gridspec
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns
    import torch
    import umap
    from numpy.random import Generator
    from PIL import Image
    from scipy.interpolate import BSpline, make_interp_spline
    from scipy.optimize import OptimizeResult, minimize_scalar
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from torch import Tensor
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
    from tqdm.auto import tqdm
    from transformers.models.auto.image_processing_auto import AutoImageProcessor
    from transformers.models.dinov2_with_registers import Dinov2WithRegistersModel

    from GaussianProxy.conf.training_conf import DataSet
    from GaussianProxy.utils.data import ImageDataset, RandomRotationSquareSymmetry
    from GaussianProxy.utils.misc import get_evenly_spaced_timesteps

    # ruff: enable: E402
    torch.set_grad_enabled(False)
    logger.info("done")

    # Process each dataset iteratively
    for ds in params.datasets:
        print()
        logger.info("#" * 60)
        logger.info(f"Processing dataset: {ds.name}")
        logger.info("#" * 60)

        this_run_save_path_this_ds = this_run_save_path / ds.name
        this_run_save_path_this_ds.mkdir(parents=True, exist_ok=True)

        # Save params.seed for reproducibility
        seed_file_path = this_run_save_path_this_ds / "seed.txt"
        with open(seed_file_path, "w") as f:
            f.write(str(params.seed))
        logger.debug(f"Saved params.seed to {seed_file_path}")
        # common rng for this run
        rng = np.random.default_rng(seed=params.seed)

        ###################################################################################################################
        ################################################ Compute encodings ################################################
        ###################################################################################################################
        encodings_filename = f"{encoding_scheme}_encodings.parquet"
        load_existing_encodings = False
        df: pd.DataFrame = pd.DataFrame()  # pylance is idiotic
        if params.recompute_encodings is not False:
            df_or_None = save_encodings(
                params.device,
                params.model_name,
                ds,
                params.use_model_preprocessor,
                this_run_save_path_this_ds,
                encodings_filename,
                params.batch_size,
                params.recompute_encodings,
            )
            if df_or_None is None:
                load_existing_encodings = True
            else:
                df = df_or_None
        else:
            load_existing_encodings = True

        if load_existing_encodings:
            logger.warning(f"=> Reusing existing encodings at {this_run_save_path_this_ds / encodings_filename}")
            df = pd.read_parquet(this_run_save_path_this_ds / encodings_filename)

        ###################################################################################################################
        ############################################### Visualize encodings ###############################################
        ###################################################################################################################
        cls_tokens = np.stack(df["encodings"].to_numpy())  # pyright: ignore[reportCallIssue, reportArgumentType]
        labels = df["labels"].to_numpy()
        file_paths = df["file_paths"]

        assert len(cls_tokens) == len(labels) == len(file_paths), (
            f"Expected same length for cls_tokens, labels and file_paths, got {len(cls_tokens)}, {len(labels)}, {len(file_paths)}"
        )
        logger.debug(
            f"cls_tokens.shape: {cls_tokens.shape}, len(labels): {len(labels)}, len(file_paths): {len(file_paths)}"
        )
        uniq_labs, counts = np.unique(labels, return_counts=True)
        logger.info(
            f"Total number of samples: {len(cls_tokens)} | labels: {uniq_labs} | counts: {counts} | size of encodings: {cls_tokens.shape[1]}"
        )

        # plot embeddings stats
        plot_histograms_embeddings_means_vars(cls_tokens, encoding_scheme, ds.name, this_run_save_path_this_ds)

        # Get sorted class names
        assert ds.dataset_params is not None, "Dataset parameters must be defined in the dataset instance"

        def get_sort_key(label):
            try:
                return ds.dataset_params.sorting_func(label)  # pyright: ignore[reportOptionalMemberAccess] # noqa: B023
            except AttributeError:
                # Wrap plain labels in a dummy class with `.name` attribute
                class LabelWrapper:
                    def __init__(self, name):
                        self.name = name

                return ds.dataset_params.sorting_func(LabelWrapper(label))  # pyright: ignore[reportOptionalMemberAccess] # noqa: B023

        sorted_unique_labels = sorted(set(labels), key=get_sort_key)

        # ensure unique labels are strings
        sorted_unique_labels = [str(label) for label in sorted_unique_labels]
        logger.warning(f"=> Using sorted unique labels: {sorted_unique_labels}")
        sorted_unique_label_times = get_evenly_spaced_timesteps(len(sorted_unique_labels))
        logger.warning(f"Sorted unique label times: {sorted_unique_label_times}")

        ### UMAP
        logger.info("=> UMAP")
        # 2D
        umap_2d_reducer = umap.UMAP(random_state=params.seed)
        umap_2d_embeddings: np.ndarray = umap_2d_reducer.fit_transform(cls_tokens)  # pyright: ignore[reportAssignmentType]
        plot_2D_embeddings(
            sorted_unique_labels,
            labels,
            umap_2d_reducer,
            umap_2d_embeddings,
            encoding_scheme,
            ds,
            rng,
            "UMAP",
            this_run_save_path_this_ds,
            f"Seed={params.seed}",
        )
        # 3D
        umap_3d_reducer = umap.UMAP(random_state=params.seed, n_components=3)
        umap_3d_embeddings: np.ndarray = umap_3d_reducer.fit_transform(cls_tokens)  # pyright: ignore[reportAssignmentType]
        plot_3D_embeddings(
            labels,
            sorted_unique_labels,
            umap_3d_embeddings,
            umap_3d_reducer,
            rng,
            this_run_save_path_this_ds,
            ds,
            "UMAP",
            encoding_scheme,
            f"Seed={params.seed}",
        )

        ### PCA
        logger.info("=> PCA")
        pca = PCA(random_state=params.seed)
        pca_embeddings = pca.fit_transform(cls_tokens)
        pca_explained_variance = pca.explained_variance_ratio_
        # 2D
        plot_2D_embeddings(
            sorted_unique_labels,
            labels,
            pca,
            pca_embeddings,
            encoding_scheme,
            ds,
            rng,
            "PCA",
            this_run_save_path_this_ds,
            f"Total explained variance: {np.sum(pca_explained_variance[:2]) * 100:.1f}%",
            (
                f"PCA 1 ({pca_explained_variance[0] * 100:.1f}% of explained variance)",
                f"PCA 2 ({pca_explained_variance[1] * 100:.1f}% of explained variance)",
            ),
        )
        # 3D
        plot_3D_embeddings(
            labels,
            sorted_unique_labels,
            pca_embeddings,
            pca,
            rng,
            this_run_save_path_this_ds,
            ds,
            "PCA",
            encoding_scheme,
            f"Seed={params.seed} | Total explained variance: {np.sum(pca_explained_variance[:3]) * 100:.1f}%",
            (
                f"PCA 1 ({pca_explained_variance[0] * 100:.1f}% of explained variance)",
                f"PCA 2 ({pca_explained_variance[1] * 100:.1f}% of explained variance)",
                f"PCA 3 ({pca_explained_variance[2] * 100:.1f}% of explained variance)",
            ),
        )

        ### LDA
        logger.info("=> LDA")
        lda = LinearDiscriminantAnalysis()
        lda_embeddings = lda.fit_transform(cls_tokens, labels)
        lda_explained_variance = lda.explained_variance_ratio_
        ## 2D
        plot_2D_embeddings(
            sorted_unique_labels,
            labels,
            lda,
            lda_embeddings,
            encoding_scheme,
            ds,
            rng,
            "LDA",
            this_run_save_path_this_ds,
            f"Total explained variance: {np.sum(lda_explained_variance[:2]) * 100:.1f}%",
            (
                f"LDA 1 ({lda_explained_variance[0] * 100:.1f}% of explained variance)",
                f"LDA 2 ({lda_explained_variance[1] * 100:.1f}% of explained variance)",
            ),
            centroids_in_base_embedding_space=lda.means_,  # pyright: ignore[reportArgumentType]
        )
        # TODO: decision boundaries
        # # 1) take your 3‐dim LDA embedding and slice to 2D
        # X2 = embeddings[:, :2]
        # y = np.array(labels)
        # # 2) fit a fresh 2D LDA (so the estimator n_features_in_ matches X2.shape[1])
        # lda_viz = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
        # lda_viz.fit(X2, y)
        # # 3) plot
        # fig, ax = plt.subplots(figsize=(10, 10))
        # plot_result(lda_viz, X2, y, ax)
        # ax.set_title(
        #     f"LDA projection of CLS tokens of {params.model_name} on {ds.name}\n"
        #     f"Total explained variance: {np.sum(explained_variance[:2]) * 100:.1f}% | decision boundaries based on separatly fitted 2D LDA"
        # )
        # ax.set_xlabel(f"LDA 1 ({explained_variance[0] * 100:.1f}% var)")
        # ax.set_ylabel(f"LDA 2 ({explained_variance[1] * 100:.1f}% var)")
        # plt.show()
        # 3D
        plot_3D_embeddings(
            labels,
            sorted_unique_labels,
            lda_embeddings,
            lda,
            rng,
            this_run_save_path_this_ds,
            ds,
            "LDA",
            encoding_scheme,
            f"Total explained variance: {np.sum(lda_explained_variance[:3]) * 100:.1f}%",
            (
                f"LDA 1 ({lda_explained_variance[0] * 100:.1f}% of explained variance)",
                f"LDA 2 ({lda_explained_variance[1] * 100:.1f}% of explained variance)",
                f"LDA 3 ({lda_explained_variance[2] * 100:.1f}% of explained variance)",
            ),
            centroids_in_base_embedding_space=lda.means_,  # pyright: ignore[reportArgumentType]
        )

        ###################################################################################################################
        ############################################# Derive continuous time ##############################################
        ###################################################################################################################
        logger.info("=> Deriving continuous time predictions from LDA embeddings")

        ### From pure proba
        logger.info("-> from pure probabilities")
        proba = lda.predict_proba(cls_tokens)
        lda_class_times = np.array(  # we just do this mapping manually to ensure the order is correct
            [sorted_unique_label_times[sorted_unique_labels.index(str(cls))] for cls in lda.classes_],  # pyright: ignore[reportGeneralTypeIssues]
            dtype=np.float32,
        )
        logger.warning(f"Derived LDA class times: {lda_class_times} from LDA classes: {lda.classes_}")
        continuous_time_predictions = proba @ lda_class_times
        logger.debug(
            f"Computed continuous time predictions from LDA probabilities, shape: {continuous_time_predictions.shape}, excerpt: {continuous_time_predictions[:5]}"
        )
        plot_histograms_continuous_time_preds(
            this_run_save_path_this_ds,
            continuous_time_predictions,
            labels,
            "LDA_probabilities",
            encoding_scheme,
            ds.name,
            True,
        )
        plot_boxplots_continuous_time_preds(
            this_run_save_path_this_ds, labels, sorted_unique_labels, continuous_time_predictions, "LDA_probabilities"
        )

        ### From decision function
        # ie signed distance to the hyperplane
        logger.info("-> from decision function")
        scores = lda.decision_function(cls_tokens)
        continuous_time_predictions = scores @ lda_class_times
        logger.debug(
            f"Computed continuous time predictions from LDA decision function, shape: {continuous_time_predictions.shape}, excerpt: {continuous_time_predictions[:5]}"
        )
        continuous_time_predictions -= continuous_time_predictions.min()
        continuous_time_predictions /= continuous_time_predictions.max()
        logger.debug("Min-max normalized continuous time predictions to [0, 1]")
        plot_histograms_continuous_time_preds(
            this_run_save_path_this_ds,
            continuous_time_predictions,
            labels,
            "LDA_decision_function",
            encoding_scheme,
            ds.name,
        )
        plot_boxplots_continuous_time_preds(
            this_run_save_path_this_ds,
            labels,
            sorted_unique_labels,
            continuous_time_predictions,
            "LDA_decision_function",
            len(df) <= 10_000,
        )

        ### From projection on spline going through class centroids
        # of base encoding space (eg DINO's CLS tokens)
        # warning: we need to sort the class centroids in the same order as the unique times/labels!
        lda_classes_in_default_order = [str(cl) for cl in lda.classes_]  # pyright: ignore[reportGeneralTypeIssues]
        lda_classes_sorting_idxes = [lda_classes_in_default_order.index(label) for label in sorted_unique_labels]
        assert np.all(
            (sorted_lda_classes := [str(lda.classes_[idx]) for idx in lda_classes_sorting_idxes])  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue]
            == sorted_unique_labels
        ), (
            f"Expected lda.classes_= {lda.classes_} sorted with {lda_classes_sorting_idxes} to be equal to {sorted_unique_labels}, got {sorted_lda_classes}"
        )
        sorted_lda_centroids: np.ndarray = lda.means_[lda_classes_sorting_idxes]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportAssignmentType]
        logger.info("-> from projection on spline going through class centroids in base encoding space")
        logger.debug(
            f"Centroids shape: {sorted_lda_centroids.shape} | true labels: {sorted_unique_labels} | true label times {sorted_unique_label_times}"
        )
        continuous_time_predictions = fit_spline_project_time_plots(
            "base embedding space",
            sorted_lda_centroids,
            sorted_unique_labels,
            sorted_unique_label_times,
            params,
            cls_tokens,
            labels,
            ds,
            rng,
            umap_2d_reducer,
            umap_2d_embeddings,
            umap_3d_reducer,
            umap_3d_embeddings,
            pca,
            pca_embeddings,
            lda,
            lda_embeddings,
            this_run_save_path_this_ds,
            encoding_scheme,
            pca_explained_variance,
            lda_explained_variance,
            df,
            "min-max",
        )
        # of LDA embeddings
        logger.info("-> from projection on spline going through class centroids in LDA encoding space")
        continuous_time_predictions = fit_spline_project_time_plots(
            "LDA embedding space",
            sorted_lda_centroids,
            sorted_unique_labels,
            sorted_unique_label_times,
            params,
            cls_tokens,
            labels,
            ds,
            rng,
            umap_2d_reducer,
            umap_2d_embeddings,
            umap_3d_reducer,
            umap_3d_embeddings,
            pca,
            pca_embeddings,
            lda,
            lda_embeddings,
            this_run_save_path_this_ds,
            encoding_scheme,
            pca_explained_variance,
            lda_explained_variance,
            df,
            "min-max",
        )

        ### Save new continuous time label
        new_ds_file_save_path = (
            Path(ds.path).parent / f"{ds.name}__continuous_time_predictions__{encoding_scheme}.parquet"
        )
        logger.info(f"Original dataset path:                        {ds.path}")
        logger.info(f"Saving continuous time predictions dataset to {new_ds_file_save_path}")

        data = []
        for sample_file_path, true_time_label, continuous_time_pred in zip(
            file_paths, labels, continuous_time_predictions, strict=True
        ):
            data.append(
                {"time": continuous_time_pred, "file_path": sample_file_path, "true_label": str(true_time_label)}
            )
        df = pd.DataFrame(data)

        if params.save_times != "no":
            write_times = True
            if new_ds_file_save_path.exists():
                if params.save_times == "no-overwrite":
                    logger.info(f"File {new_ds_file_save_path} already exists, skipping saving")
                    write_times = False
                elif params.save_times == "overwrite":
                    logger.info(f"File {new_ds_file_save_path} already exists, overwriting")
                    write_times = True
                elif params.save_times == "ask-before-overwrite":
                    inpt = input(f"Delete existing file at {new_ds_file_save_path}? (y/n)")
                    print("", flush=True)
                    if inpt == "y":
                        write_times = True
                    else:
                        logger.info("Refusing to continue")
                        write_times = False
                else:
                    raise ValueError(f"Unknown params.save_times value: {params.save_times}")
            if write_times:
                df.to_parquet(new_ds_file_save_path, index=False)
                logger.info(f"Saved continuous time predictions dataset to {new_ds_file_save_path}")
        else:
            logger.warning("No times saving")
