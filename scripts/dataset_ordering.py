# pylint: disable=too-many-lines
# pylint: disable=possibly-used-before-assignment
# pylint: disable=logging-fstring-interpolation
# pylint: disable=missing-module-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring

# Most imports are made after the parameters printing section because Python imports are just so fucking slow
from __future__ import annotations

import logging
import pickle
import random
import sys
from pathlib import Path

import attrs
import colorlog
from rich.traceback import install


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
        prefetch_factor=2,
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
    recompute_encodings: Literal["no-overwrite", "force-overwrite"],
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
    model = AutoModel.from_pretrained(model_name).to(device)  # pyright: ignore[reportArgumentType]

    # Run
    rows: list[dict] = []

    for batch in tqdm(iter(dataloader), total=len(dataloader), desc="Processing batches"):
        # `tensors` is only used if use_model_preprocessor is `False`!
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

        # labels
        labels: list[int] | list[str] = [dataset.dataset_params.key_transform(p.parent.name) for p in paths]

        # save each row of data to rows list
        for this_cls_token, this_label, this_path in zip(cls_tokens, labels, paths, strict=True):
            row_data = {
                "encodings": this_cls_token,
                "labels": this_label,
                "file_paths": str(this_path),
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
    projector_embeddings_test: np.ndarray | None = None,
    labels_test: np.ndarray | None = None,
):
    plt.figure(figsize=(10, 10))
    # Use a sequential color palette with as many colors as unique labels
    palette = sns.color_palette("viridis", n_colors=len(unique_labels))
    label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

    # concatenate test data with main one if present
    if projector_embeddings_test is not None or labels_test is not None:
        assert projector_embeddings_test is not None and labels_test is not None, (
            f"Both projector_embeddings_test and labels_test must be provided if one of them is provided, got {type(projector_embeddings_test)} and {type(labels_test)}"
        )
        projector_embeddings = np.concatenate((projector_embeddings, projector_embeddings_test), axis=0)
        labels = np.concatenate((labels, labels_test), axis=0)
        is_test_data = np.array([False] * len(labels) + [True] * len(labels_test), dtype=bool)
    else:
        is_test_data = np.array([False] * len(labels), dtype=bool)

    # checks
    assert len(labels) == len(projector_embeddings), (
        f"Expected labels and embeddings to have the same length, got {len(labels)} and {projector_embeddings.shape}"
    )

    # Make the plot order random
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    shuffled_embeddings = projector_embeddings[indices]
    shuffled_labels = [labels[i] for i in indices]
    shuffled_colors = [label_to_color[str(label)] for label in shuffled_labels]
    is_test_data = is_test_data[indices]

    # 1. Plot points
    plt.scatter(
        shuffled_embeddings[~is_test_data, 0], shuffled_embeddings[~is_test_data, 1], c=shuffled_colors, s=10, alpha=0.5
    )
    if np.any(is_test_data):
        plt.scatter(
            shuffled_embeddings[is_test_data, 0],
            shuffled_embeddings[is_test_data, 1],
            c=np.array(shuffled_colors)[is_test_data],
            s=10,
            alpha=0.5,
            marker="^",
            edgecolor="black",
        )
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()

    plt.gca().set_aspect("equal", "datalim")
    plt.suptitle(f"{viz_name.split('_')[0]} projection of CLS tokens of {encoding_scheme} on {dataset.name}")
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
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

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
    projector_embeddings_test: np.ndarray | None = None,
    labels_test: np.ndarray | None = None,
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

    # concatenate test data with main one if present
    if projector_embeddings_test is not None or labels_test is not None:
        assert projector_embeddings_test is not None and labels_test is not None, (
            f"Both projector_embeddings_test and labels_test must be provided if one of them is provided, got {type(projector_embeddings_test)} and {type(labels_test)}"
        )
        projector_embeddings = np.concatenate((projector_embeddings, projector_embeddings_test), axis=0)
        labels = np.concatenate((labels, labels_test), axis=0)
        is_test_data = np.array([False] * len(labels) + [True] * len(labels_test), dtype=bool)
    else:
        is_test_data = np.array([False] * len(labels), dtype=bool)

    # Make the plot order random
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    shuffled_projector_embeddings = projector_embeddings[indices]
    shuffled_labels = [labels[i] for i in indices]
    is_test_data = is_test_data[indices]

    if xyz_labels is None:
        xyz_labels = (f"{viz_name} 1", f"{viz_name} 2", f"{viz_name} 3")

    title = f"3D {viz_name.split('_')[0]} projection of CLS tokens of {encoding_scheme} from {dataset.name}"
    if subtitle is not None:
        title += f"<br><sub>{subtitle}</sub>"

    # 1. Plot points
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
        symbol=np.where(is_test_data, "diamond", "circle"),
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


def project_to_time(
    points: np.ndarray, spline_values: np.ndarray, times_of_spline_evaluation: np.ndarray
) -> np.ndarray:
    """
    Projection of points to time using the spline already computed values at `times_of_spline_evaluation`.
    Simply returns the closest time value for each point.
    """
    assert times_of_spline_evaluation.ndim == 1
    assert spline_values.ndim == points.ndim == 2, f"Expected 2D arrays, got {spline_values.ndim}D and {points.ndim}D"
    assert spline_values.shape[1] == points.shape[1], (
        f"Expected same number of features, got {spline_values.shape[1]} and {points.shape[1]}"
    )
    # torch cdist is much faster than scipy cdist; move to GPU if it's still too slow
    dist_matrix = cdist(
        torch.tensor(points, dtype=torch.float32), torch.tensor(spline_values, dtype=torch.float32), p=2
    )
    dist_matrix = dist_matrix.numpy()
    assert dist_matrix.shape == (points.shape[0], spline_values.shape[0])
    closest_indices = np.argmin(dist_matrix, axis=1)
    assert closest_indices.shape == (points.shape[0],)
    times = times_of_spline_evaluation[closest_indices]
    assert times.shape == (points.shape[0],)
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
    possible_diffs = sorted(set(a - b for a in sorted_unique_label_times for b in sorted_unique_label_times))
    changes_count_by_dist = dict.fromkeys(possible_diffs, 0)

    for true_label, time in zip(labels, continuous_times, strict=True):
        time: float
        true_label_time = sorted_unique_label_times[sorted_unique_labels.index(str(true_label))]
        dist_to_true_label = abs(true_label_time - time)
        dists_to_true_label.append(dist_to_true_label)

        closest_label_time = min(sorted_unique_label_times, key=lambda x: abs(x - time))
        closest_label = sorted_unique_labels[sorted_unique_label_times.index(closest_label_time)]

        if closest_label != str(true_label):
            nb_changes += 1
            changes_count_from_true_label[str(true_label)] += 1

        changes_count_by_dist[true_label_time - closest_label_time] += 1

    logger.debug(f"Number of changes: {nb_changes} out of {len(labels)} ({nb_changes / len(labels) * 100:.2f}%)")
    logger.debug(f"Changes count from true label: {changes_count_from_true_label}")

    n_labels = len(sorted_unique_labels)
    n_cols = min(4, n_labels)
    n_rows_middle_plot = (n_labels + n_cols - 1) // n_cols
    n_rows = n_rows_middle_plot + 2  # top hist, per-label hists, changes hist

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows_middle_plot + 6))
    gs = gridspec.GridSpec(nrows=n_rows, ncols=n_cols, height_ratios=[1.2] + [1] * n_rows_middle_plot + [1.2])

    max_dist_to_true_label = max(dists_to_true_label)

    # Top: combined histogram spanning all columns
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.hist(dists_to_true_label, bins=256)
    ax_top.set_title("Histogram of distances between continuous time pred and true original label")
    ax_top.set_xlabel("Distance to true label")
    ax_top.set_ylabel("Count")
    ax_top.set_xlim(0, max_dist_to_true_label)

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
        ax.set_xlim(0, max_dist_to_true_label)

    # Histogram of changes_count_by_dist
    ax_changes = fig.add_subplot(gs[-1, :])
    diffs = list(changes_count_by_dist.keys())
    counts = list(changes_count_by_dist.values())
    sns.barplot(x=diffs, y=counts, ax=ax_changes)
    # Limit the number of x-tick labels for readability
    max_ticks = 10
    # show up to max_ticks evenly spaced labels at bar positions
    n = len(diffs)
    step = max(1, n // max_ticks)
    positions = diffs[::step]
    xticklabels = [f"{p:.2g}" for p in positions]
    ax_changes.set_xticks(positions)
    ax_changes.set_xticklabels(xticklabels, rotation=45, ha="right")

    ax_changes.set_title("Histogram of distances between true original label and true label closest to time prediction")
    ax_changes.set_xlabel("(time of true original label) - (time of new closest label)")
    ax_changes.set_ylabel("Count")

    plt.suptitle(f"Distances to true label histograms for {dataset_name} with {viz_name}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_path = base_save_path / f"continuous_time_predictions_distances_to_true_labels_{viz_name}_histogram.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.debug(f"Saved histograms of distances to true labels to {save_path}")


def get_stratified_random_indices(labels: np.ndarray, n_per_class: int, rng: np.random.Generator):
    idxs = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_idxs = np.where(labels == label)[0]
        n = min(n_per_class, len(label_idxs))
        idxs.extend(rng.choice(label_idxs, size=n, replace=False))
    return np.array(idxs)


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
    nb_times_spline_eval: int,
    labels_test: np.ndarray | None = None,
    umap_2d_embeddings_test: np.ndarray | None = None,
    umap_3d_embeddings_test: np.ndarray | None = None,
    pca_embeddings_test: np.ndarray | None = None,
    lda_embeddings_test: np.ndarray | None = None,
):
    # Checks
    vals_check = [
        labels_test is None,
        umap_2d_embeddings_test is None,
        umap_3d_embeddings_test is None,
        pca_embeddings_test is None,
        lda_embeddings_test is None,
    ]
    if any(vals_check):
        assert all(vals_check), f"Expected all test embeddings to be None, got {vals_check}"

    # init
    continuous_time_predictions_test = None

    # Compute spline
    logger.debug(f"Fitting interpolating spline through class centroids of {projection_type} of {encoding_scheme}")
    if projection_type == "base embedding space":
        centroids_to_use_to_fit = sorted_centroids_in_base_embedding_space
        embeddings_to_project = cls_tokens
    elif projection_type == "LDA embedding space":
        centroids_in_lda_embedding_space = lda.transform(sorted_centroids_in_base_embedding_space)
        centroids_to_use_to_fit = centroids_in_lda_embedding_space
        embeddings_to_project = lda_embeddings
    else:
        raise ValueError(
            f"Unknown projection type: {projection_type}. Expected 'base embedding space' or 'LDA embedding space'."
        )
    logger.debug(
        f"Centroids shape: {centroids_to_use_to_fit.shape} | true labels: {sorted_unique_labels} | true label times {sorted_unique_label_times}"
    )
    # Fit spline
    spline = make_interp_spline(sorted_unique_label_times, centroids_to_use_to_fit)
    # Evaluate spline
    assert (sorted_unique_label_times[0], sorted_unique_label_times[-1]) == (0, 1)
    t_min, t_max = -params.spline_continuation_range, 1 + params.spline_continuation_range
    logger.warning(
        f"Using additional range {params.spline_continuation_range} for t_min, t_max = {t_min}, {t_max} parametrization of the spline with {nb_times_spline_eval} evaluation points"
    )
    times_to_eval_spline = np.linspace(t_min, t_max, nb_times_spline_eval)
    spline_values = spline(times_to_eval_spline)
    # Project embeddings on spline
    logger.debug(
        f"projecting embeddings of shape {embeddings_to_project.shape} on spline of shape {spline_values.shape}"
    )
    continuous_time_predictions = project_to_time(embeddings_to_project, spline_values, times_to_eval_spline)
    if lda_embeddings_test is not None:
        assert projection_type == "LDA embedding space", (
            f"Expected projection_type to be 'LDA embedding space', got {projection_type}"
        )
        continuous_time_predictions_test = project_to_time(lda_embeddings_test, spline_values, times_to_eval_spline)
    logger.debug(
        f"Computed continuous time predictions from spline projection, shape: {continuous_time_predictions.shape}, excerpt: {continuous_time_predictions[:5]}"
    )
    # Plot spline and projections along with base embeddings
    random_idx_to_plot_projections = get_stratified_random_indices(labels, 10, rng)
    projection_pairs = (
        embeddings_to_project[random_idx_to_plot_projections],
        spline(continuous_time_predictions[random_idx_to_plot_projections]),
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
            f"Seed={params.seed} | params.spline_continuation_range={params.spline_continuation_range}",
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
        projector_embeddings_test=lda_embeddings_test,
        labels_test=labels_test,
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
            f"Seed={params.seed} | params.spline_continuation_range={params.spline_continuation_range}",
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
            f"Total explained variance: {np.sum(pca_explained_variance[:3]) * 100:.1f}% | seed={params.seed} | params.spline_continuation_range={params.spline_continuation_range}",
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
        f"Total explained variance: {np.sum(lda_explained_variance[:3]) * 100:.1f}% | params.spline_continuation_range={params.spline_continuation_range}",
        (
            f"LDA 1 ({lda_explained_variance[0] * 100:.1f}% of explained variance)",
            f"LDA 2 ({lda_explained_variance[1] * 100:.1f}% of explained variance)",
            f"LDA 3 ({lda_explained_variance[2] * 100:.1f}% of explained variance)",
        ),
        spline_values,
        projection_pairs,
        sorted_centroids_in_base_embedding_space,  # pyright: ignore[reportArgumentType]
        projection_type,
        projector_embeddings_test=lda_embeddings_test,
        labels_test=labels_test,
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
    if continuous_time_predictions_test is not None:
        assert labels_test is not None
        plot_histograms_continuous_time_preds(
            this_run_save_path_this_ds,
            continuous_time_predictions_test,
            labels_test,
            vizes_name_suffix + "_test",
            encoding_scheme,
            ds.name,
        )
        plot_boxplots_continuous_time_preds(
            this_run_save_path_this_ds,
            labels_test,
            sorted_unique_labels,
            continuous_time_predictions_test,
            vizes_name_suffix + "_test",
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
    if continuous_time_predictions_test is not None:
        assert labels_test is not None
        plot_histograms_distances_to_true_labels(
            sorted_unique_label_times,
            sorted_unique_labels,
            continuous_time_predictions_test,
            ds.name,
            this_run_save_path_this_ds,
            labels_test,
            vizes_name_suffix + "_test",
        )

    # Concatenate train and test predictions
    if continuous_time_predictions_test is not None:
        assert labels_test is not None
        logger.info(
            f"Concatenating continuous time predictions for train set (shape: {continuous_time_predictions.shape}) and test set (shape: {continuous_time_predictions_test.shape})"
        )
        # TODO: refit and refit and repredict on full data instead...
        continuous_time_predictions = np.concatenate(
            [continuous_time_predictions, continuous_time_predictions_test], axis=0
        )
        logger.info(f"Concatenated continuous time predictions shape: {continuous_time_predictions.shape}")
        train_test_labels = np.array(["train"] * len(labels) + ["test"] * len(labels_test))
        logger.debug(f"Generated train/test split labels of shape: {train_test_labels.shape}")
        logger.info(
            f"Concatenating labels for train set (shape: {labels.shape}) and test set (shape: {labels_test.shape})"
        )
        labels = np.concatenate([labels, labels_test], axis=0)
        logger.info(f"Concatenated labels shape: {labels.shape}")
    else:
        train_test_labels = None

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

    return continuous_time_predictions, labels, train_test_labels


def get_sort_func(ds, label):
    """
    Gets the right sorting function for the given dataset and label.
    """
    import pandas as pd

    class LabelWrapper:
        def __init__(self, name):
            self.name = name

    if isinstance(label, pd.Index):
        index_list = label.tolist()
        try:
            sorted_index = sorted(index_list, key=ds.dataset_params.sorting_func)  # pyright: ignore[reportOptionalMemberAccess]
        except AttributeError:
            # Wrap plain labels in a dummy class with `.name` attribute
            sorted_index = sorted(index_list, key=lambda x: ds.dataset_params.sorting_func(LabelWrapper(x)))  # pyright: ignore[reportOptionalMemberAccess]
        return sorted_index
    else:
        try:
            return ds.dataset_params.sorting_func(label)  # pyright: ignore[reportOptionalMemberAccess] # noqa: B023
        except AttributeError:
            # Wrap plain labels in a dummy class with `.name` attribute
            return ds.dataset_params.sorting_func(LabelWrapper(label))  # pyright: ignore[reportOptionalMemberAccess] # noqa: B023


@attrs.define
class Params:
    """
    - `test_split_frac`: if None, no test split is performed, otherwise the sampling is stratified by labels
    - `spline_continuation_range`: time range to use for t_min, t_max parametrization of the spline
    beyond the [0;1] time range defined by the extremal class centroids.
    - `border_centroids`: if True, 2 additional extremal centroids are computed: the centroids of the points
    which projection on the spline is the first or last class centroid. These point act as "extremal" subclasses,
    and are used to continue the spline beyond the [0;1] time range in a "principled" way. The spline is then continued on a
    `spline_continuation_range` range before/after these *new* extremal centroids.

    """

    datasets: list[DataSet]
    device: str
    model_name: str
    batch_size: int
    use_model_preprocessor: bool
    recompute_encodings: Literal["force-overwrite", "no-overwrite", "no"]
    save_times: Literal["no-overwrite", "overwrite", "ask-before-overwrite", "no"]
    seed: int
    spline_continuation_range: float
    nb_times_spline_eval: int
    test_split_frac: float | None = None


if __name__ == "__main__":
    ###################################################################################################################
    ################################################### Parameters ####################################################
    ###################################################################################################################
    install(show_locals=True, width=200)
    sys.path.append(".")
    # Attention: we *might or might not* use our datasets' pipeline as DINO has its own preprocessing pipeline.
    # ruff: noqa: F401
    # pylint: disable=unused-import
    from my_conf.dataset.BBBC021.BBBC021_196_docetaxel_inference import (
        BBBC021_196_docetaxel_inference as bbbc021_ds,
    )
    from my_conf.dataset.biotine.biotine_png_128_inference import dataset as biotine_ds
    from my_conf.dataset.ChromaLive6h.chromalive6h_3ch_png_inference import dataset as chromalive_ds
    from my_conf.dataset.diabetic_retinopathy.diabetic_retinopathy_inference import (
        diabetic_retinopathy_inference as diabetic_retinopathy_ds,
    )
    from my_conf.dataset.ependymal_context.ependymal_context_inference import dataset as ependymal_context_ds
    from my_conf.dataset.ependymal_cutout.ependymal_cutout_inference import dataset as ependymal_cutout_ds
    from my_conf.dataset.Jurkat.Jurkat_inference import Jurkat_inference as jurkat_ds
    from my_conf.dataset.NASH_fibrosis.NASH_fibrosis_inference import dataset as NASH_fibrosis_ds
    from my_conf.dataset.NASH_steatosis.NASH_steatosis_inference import dataset as NASH_steatosis_ds
    # pylint: enable=unused-import
    # ruff: enable=F401

    # fmt: off
    params = Params(
        datasets                  = [bbbc021_ds, biotine_ds, chromalive_ds, diabetic_retinopathy_ds, ependymal_context_ds, ependymal_cutout_ds, jurkat_ds, NASH_fibrosis_ds, NASH_steatosis_ds],
        device                    = "cuda:2",
        model_name                = "facebook/dinov2-with-registers-giant",
        batch_size                = 256,
        use_model_preprocessor    = False,
        recompute_encodings       = "no",
        save_times                = "overwrite",
        seed                      = random.randint(0, 2**32 - 1),
        spline_continuation_range = 0.1,
        nb_times_spline_eval      = 10_000,
        test_split_frac           = 0.1,
    )
    # fmt: on

    ###################################################################################################################
    ##################################################### Launch ######################################################
    ###################################################################################################################
    base_save_dir = Path("/projects/static2dynamic/Thomas/ordering_datasets")
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
    from functools import partial
    from typing import Literal, cast

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
    from scipy.interpolate import make_interp_spline
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import train_test_split
    from torch import Tensor, cdist
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
    from tqdm.auto import tqdm
    from transformers.models.auto.image_processing_auto import AutoImageProcessor
    from transformers.models.auto.modeling_auto import AutoModel

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
        if params.recompute_encodings != "no":
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

        df_test: pd.DataFrame | None = None
        if params.test_split_frac is not None:
            # stratify the sampling by label
            splits = train_test_split(
                df,
                test_size=params.test_split_frac,
                random_state=params.seed,
                stratify=df["labels"],
            )
            df: pd.DataFrame = splits[0]
            df_test = cast(pd.DataFrame, splits[1])
            proportions_to_print = (
                pd.concat(
                    [
                        df["labels"].value_counts(normalize=True),
                        df_test["labels"].value_counts(normalize=True),
                    ],
                    axis=1,
                )
                .set_axis(["train", "test"], axis=1)
                .sort_index(key=partial(get_sort_func, ds))  # pyright: ignore[reportArgumentType]
                * 100
            )
            logger.info(
                f"Created stratified train/test splits with {len(df)}/{len(df_test)} samples; normalized composition (%):\n{proportions_to_print}"
            )
        else:
            logger.info("No test split performed, using all samples for fitting")

        ###################################################################################################################
        ############################################### Visualize encodings ###############################################
        ###################################################################################################################
        cls_tokens = np.stack(df["encodings"].to_numpy())  # pyright: ignore[reportCallIssue, reportArgumentType]
        labels = df["labels"].to_numpy()
        file_paths = df["file_paths"]

        if df_test is not None:
            cls_tokens_test = np.stack(df_test["encodings"].to_numpy())  # pyright: ignore[reportCallIssue, reportArgumentType]
            labels_test = df_test["labels"].to_numpy()
            file_paths_test = df_test["file_paths"]
        else:
            cls_tokens_test, labels_test, file_paths_test = None, None, None

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

        sorted_unique_labels = sorted(set(labels), key=partial(get_sort_func, ds))

        # ensure unique labels are strings
        sorted_unique_labels = [str(label) for label in sorted_unique_labels]
        logger.warning(f"=> Using sorted unique labels: {sorted_unique_labels}")
        sorted_unique_label_times = get_evenly_spaced_timesteps(len(sorted_unique_labels))
        logger.warning(f"Sorted unique label times: {sorted_unique_label_times}")

        # get class centroids in base embedding space
        class_means = np.array(
            [cls_tokens[labels.astype("str") == label].mean(axis=0) for label in sorted_unique_labels]
        )

        ### UMAP
        logger.info("=> UMAP")
        # 2D
        umap_2d_reducer = umap.UMAP(random_state=params.seed)
        umap_2d_embeddings: np.ndarray = umap_2d_reducer.fit_transform(cls_tokens)  # pyright: ignore[reportAssignmentType]
        umap_2d_embeddings_test: np.ndarray = (
            umap_2d_reducer.transform(cls_tokens_test) if cls_tokens_test is not None else None  # pyright: ignore[reportAssignmentType]
        )
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
            projector_embeddings_test=umap_2d_embeddings_test,
            labels_test=labels_test,
            centroids_in_base_embedding_space=class_means,
        )
        # 3D
        umap_3d_reducer = umap.UMAP(random_state=params.seed, n_components=3)
        umap_3d_embeddings: np.ndarray = umap_3d_reducer.fit_transform(cls_tokens)  # pyright: ignore[reportAssignmentType]
        umap_3d_embeddings_test: np.ndarray = (
            umap_3d_reducer.transform(cls_tokens_test) if cls_tokens_test is not None else None  # pyright: ignore[reportAssignmentType]
        )
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
            projector_embeddings_test=umap_3d_embeddings_test,
            labels_test=labels_test,
            centroids_in_base_embedding_space=class_means,
        )

        ### PCA
        logger.info("=> PCA")
        pca = PCA(random_state=params.seed)
        pca_embeddings = pca.fit_transform(cls_tokens)
        pca_explained_variance = pca.explained_variance_ratio_
        pca_embeddings_test = pca.transform(cls_tokens_test) if cls_tokens_test is not None else None
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
            projector_embeddings_test=pca_embeddings_test,
            labels_test=labels_test,
            centroids_in_base_embedding_space=class_means,
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
            projector_embeddings_test=pca_embeddings_test,
            labels_test=labels_test,
            centroids_in_base_embedding_space=class_means,
        )

        ### LDA
        logger.info("=> LDA")
        lda = LinearDiscriminantAnalysis()
        lda_embeddings = lda.fit_transform(cls_tokens, labels)
        # save the LDA model
        with open(this_run_save_path_this_ds / "lda.pickle", "wb") as f:
            pickle.dump(lda, f)
        logger.info(f"Saved LDA model to {this_run_save_path_this_ds / 'lda.pickle'}")
        lda_explained_variance = lda.explained_variance_ratio_
        lda_embeddings_test = lda.transform(cls_tokens_test) if cls_tokens_test is not None else None
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
            projector_embeddings_test=lda_embeddings_test,
            labels_test=labels_test,
        )
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
            projector_embeddings_test=lda_embeddings_test,
            labels_test=labels_test,
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
        # of base encoding space (eg DINO's CLS tokens)
        logger.info("-> from projection on spline going through class centroids in base encoding space")
        logger.debug(
            f"Centroids shape: {sorted_lda_centroids.shape} | true labels: {sorted_unique_labels} | true label times {sorted_unique_label_times}"
        )
        _ = fit_spline_project_time_plots(
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
            params.nb_times_spline_eval,
        )
        # of LDA embeddings
        logger.info("-> from projection on spline going through class centroids in LDA encoding space")
        continuous_time_predictions, labels, train_test_labels = fit_spline_project_time_plots(
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
            params.nb_times_spline_eval,
            labels_test,
            umap_2d_embeddings_test,
            umap_3d_embeddings_test,
            pca_embeddings_test,
            lda_embeddings_test,
        )

        ### Save new continuous time label
        new_ds_file_save_path = (
            Path(ds.path).parent / f"{ds.name}__continuous_time_predictions__{encoding_scheme}.parquet"
        )

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
                logger.info(f"Original dataset path:                        {ds.path}")
                logger.info(f"Saving continuous time predictions dataset to {new_ds_file_save_path}")

                data = []

                if params.test_split_frac is not None:
                    # still nee to concatenate train and test for file_paths only
                    assert len(continuous_time_predictions) == len(labels) > len(file_paths), (
                        f"Expected len(continuous_time_predictions) == len(labels) > len(file_paths), got {len(continuous_time_predictions)}, {len(labels)}, {len(file_paths)}"
                    )
                    file_paths = pd.concat([file_paths, file_paths_test], axis=0)

                for sample_file_path, true_time_label, continuous_time_pred in zip(
                    file_paths, labels, continuous_time_predictions, strict=True
                ):
                    data.append(
                        {
                            "time": continuous_time_pred,
                            "file_path": sample_file_path,
                            "true_label": str(true_time_label),
                        }
                    )
                continuous_time_predictions_df = pd.DataFrame(data)
                continuous_time_predictions_df.to_parquet(new_ds_file_save_path, index=False)
                logger.info(f"Saved continuous time predictions dataset to {new_ds_file_save_path}")

                if train_test_labels is not None:
                    # train/test split is redone each time this script is run, so save it each time too, with "encoding scheme" info
                    # also, save it in a separate file because this info has no relation to training
                    data = []
                    for sample_file_path, train_test_label in zip(file_paths, train_test_labels, strict=True):
                        data.append(
                            {
                                "file_path": sample_file_path,
                                "train_test_label": train_test_label,
                            }
                        )
                    train_test_labels_df = pd.DataFrame(data)
                    save_path = new_ds_file_save_path.with_name(
                        f"{ds.name}__train_test_labels__{encoding_scheme}.parquet"
                    )
                    train_test_labels_df.to_parquet(save_path)
                    logger.info(f"Saved train/test split labels to {save_path}")
        else:
            logger.warning("No times saving")
