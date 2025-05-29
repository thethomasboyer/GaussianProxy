# pylint: disable=possibly-used-before-assignment

# Most imports are made after the parameters printing section because Python imports are just so fucking slow
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Literal

from rich.traceback import install

sys.path.append(".")
install(show_locals=True, locals_max_string=20)


#######################################################################################################################
######################################################## Utils ########################################################
#######################################################################################################################
def adapt_dataset_get_dataloader(dataset: DataSet, batch_size: int):
    # Checks
    assert dataset.dataset_params is not None, "Dataset parameters must be defined in the dataset instance"

    print("Base dataset:", dataset)

    # Data transforms
    # Globally speaking, we would like to predict the time using the *same transformations* than those used in the data loading of the generative model.
    # The (random) augmentations however must be discarded!
    #
    # *But* we also would like to use DINO's preprocessor...
    transforms_to_remove = [RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotationSquareSymmetry]
    print(f"\nRemoving transforms:\n{['.'.join([c.__module__, c.__name__]) for c in transforms_to_remove]}")

    used_transforms = Compose(
        [t for t in dataset.transforms.transforms if not any(isinstance(t, tr) for tr in transforms_to_remove)]
    )
    dataset.transforms = used_transforms
    print(f"\nNow using transforms: {used_transforms}")

    all_samples = list(Path(dataset.path).rglob(f"*.{dataset.dataset_params.file_extension}"))
    ds = ImageDataset(all_samples, dataset.transforms, dataset.expected_initial_data_range)
    print("\nInstantiated dataset:", ds)

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

    ds.__getitem__ = getitem_with_name.__get__(ds, type(ds))  # pylint: disable=no-value-for-parameter
    ds.__getitems__ = getitems_with_names.__get__(ds, type(ds))  # pylint: disable=no-value-for-parameter

    # Dataloader
    dl = DataLoader(
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
            print(
                f"Encodings already exist at {base_save_path / save_name}. Skipping recomputation since recompute_encodings is set to 'no-overwrite'"
            )
            return
        else:
            (base_save_path / save_name).unlink()
    base_save_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving new encodings to {base_save_path / save_name}")
    # Dataloader
    dataloader = adapt_dataset_get_dataloader(dataset, batch_size)

    # Model
    print(f"Loading model {model_name} on device {device}...")
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
        labels = [dataset.dataset_params.key_transform(p.parent.stem) for p in paths]

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
    print(f"Saved data to {base_save_path / save_name}")

    # smol print
    print("DataFrame head:")
    print(df.head())

    # return DataFrame
    return df


def plot_2D_embeddings(
    unique_labels: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    model_name: str,
    dataset: DataSet,
    rng: Generator,
    viz_name: str,
    base_save_path: Path,
    subtitle: str | None = None,
    xy_labels: tuple[str, str] | None = None,
):
    plt.figure(figsize=(10, 10))
    # Use a sequential color palette with as many colors as unique labels
    palette = sns.color_palette("viridis", n_colors=len(unique_labels))
    label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

    # Make the plot order random
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    shuffled_embeddings = embeddings[indices]
    shuffled_labels = [labels[i] for i in indices]
    shuffled_colors = [label_to_color[str(label)] for label in shuffled_labels]

    plt.scatter(shuffled_embeddings[:, 0], shuffled_embeddings[:, 1], c=shuffled_colors, s=10, alpha=0.5)
    plt.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=label_to_color[label], label=str(label), markersize=8
            )
            for label in unique_labels
        ]
    )
    plt.gca().set_aspect("equal", "datalim")
    plt.suptitle(f"{viz_name} projection of CLS tokens of {model_name} on {dataset.name} dataset")
    if subtitle is not None:
        plt.title(subtitle)
    if xy_labels is not None:
        plt.xlabel(xy_labels[0])
        plt.ylabel(xy_labels[1])
    else:
        plt.xlabel(f"{viz_name} 1")
        plt.ylabel(f"{viz_name} 2")

    save_path = base_save_path / f"2d_{viz_name.lower()}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved 2D plot to {save_path}")


def plot_histograms_continuous_time_preds(
    base_save_path: Path,
    continuous_time_predictions: np.ndarray,
    labels: np.ndarray,
    basename: str,
    y_log_scale: bool = False,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Top: combined histogram
    sns.histplot(x=continuous_time_predictions, hue=labels, bins=100, palette="viridis", multiple="stack", ax=ax1)
    ax1.set_title("Histogram of continuous time predictions")
    ax1.set_xlabel("Continuous time prediction")
    if y_log_scale:
        ax1.set_yscale("log")
        ax1.set_ylabel("Count (log scale)")
    else:
        ax1.set_ylabel("Count")

    # Bottom: histogram with hue by labels
    sns.histplot(x=continuous_time_predictions, hue=labels, bins=100, alpha=0.5, palette="viridis", ax=ax2)
    ax2.set_title("Histogram of continuous time predictions colored by true labels")
    ax2.set_xlabel("Continuous time prediction")
    if y_log_scale:
        ax2.set_yscale("log")
        ax2.set_ylabel("Count (log scale)")
    else:
        ax2.set_ylabel("Count")
    ax2.get_legend().set_title("True label")

    plt.suptitle(f"Continuous time predictions histogram for {basename}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_path = base_save_path / f"{basename}_continuous_time_predictions_histogram.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved histogram of continuous time predictions to {save_path}")


def plot_boxplots_continuous_time_preds(
    base_save_path: Path,
    labels: np.ndarray,
    continuous_time_predictions: np.ndarray,
    basename: str,
    plot_swarmplot: bool = False,
):
    if plot_swarmplot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    else:
        fig, ax1 = plt.subplots(figsize=(14, 6))

    # Subplot 1: Boxplot and Stripplot
    sns.boxplot(
        x=labels,
        y=continuous_time_predictions,
        palette="tab10",
        hue=labels,
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

    save_path = base_save_path / f"{basename}_continuous_time_predictions_boxplots.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved boxplots of continuous time predictions to {save_path}")


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
    spline_raw_values: np.ndarray | None = None,
    projection_pairs: tuple[np.ndarray, np.ndarray] | None = None,
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

    title = f"3D {viz_name} projection of CLS tokens of {encoding_scheme} from {dataset.name}"
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
        opacity=0.4,
    )
    fig.update_traces(marker=dict(size=2))
    # Enforce 1:1:1 aspect ratio
    fig.update_scenes(aspectmode="cube")

    # 2. Add time centroids if projector is LDA
    if isinstance(projector, LinearDiscriminantAnalysis):
        centroids_lda_embeddings = projector.transform(projector.means_)  # pyright: ignore[reportArgumentType]
        fig.add_trace(
            go.Scatter3d(
                x=centroids_lda_embeddings[:, 0],
                y=centroids_lda_embeddings[:, 1],
                z=centroids_lda_embeddings[:, 2],
                mode="markers",
                marker=dict(symbol="cross", size=5, color="darkgoldenrod"),
                name="centroids",
            )
        )

    # 3. Plot spline values
    if spline_raw_values is not None:
        spline_lda_emb: np.ndarray = projector.transform(spline_raw_values)  # pyright: ignore[reportAssignmentType]
        fig.add_trace(
            go.Scatter3d(
                x=spline_lda_emb[:, 0],
                y=spline_lda_emb[:, 1],
                z=spline_lda_emb[:, 2],
                mode="lines",
                line=dict(color="red", width=3),
                name="centroid-spline",
            )
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
        source_coord_lda_emb: np.ndarray = projector.transform(source_coord)  # pyright: ignore[reportAssignmentType]
        projected_time_coord_lda_emb: np.ndarray = projector.transform(projected_coord)  # pyright: ignore[reportAssignmentType]
        for idx in range(len(source_coord)):
            fig.add_trace(
                go.Scatter3d(
                    x=[source_coord_lda_emb[idx, 0], projected_time_coord_lda_emb[idx, 0]],
                    y=[source_coord_lda_emb[idx, 1], projected_time_coord_lda_emb[idx, 1]],
                    z=[source_coord_lda_emb[idx, 2], projected_time_coord_lda_emb[idx, 2]],
                    mode="lines",
                    line=dict(color="black", width=3),
                    legendgroup="Projection Lines",
                    name="Projection Line" if idx == 0 else None,
                    showlegend=(idx == 0),
                )
            )

    # 5. Save and return
    save_path = base_save_path / f"3d_{viz_name}.html"
    fig.write_html(save_path, auto_open=False)
    print(f"Saved interactive 3D plot to {save_path}")

    png_save_path = save_path.with_suffix(".png")
    fig.update_layout(width=1400, height=1000)
    fig.write_image(png_save_path, scale=4)
    print(f"Saved 3D plot to {png_save_path}")


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
    basename: str,
    base_save_path: Path,
    labels: np.ndarray,
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

    print(f"Number of changes: {nb_changes} out of {len(labels)} ({nb_changes / len(labels) * 100:.2f}%)")
    print(f"Changes count from true label: {changes_count_from_true_label}")
    print(
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

    plt.suptitle(f"Distances to true label histograms for {basename}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_path = base_save_path / f"{basename}_distances_to_true_labels_histogram.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved histogram of distances to true labels to {save_path}")


if __name__ == "__main__":
    ###################################################################################################################
    ################################################### Parameters ####################################################
    ###################################################################################################################
    # Attention: we *might or might not* use our datasets' pipeline as DINO has its own preprocessing pipeline.
    from my_conf.dataset.BBBC021_196_docetaxel_inference import (
        BBBC021_196_docetaxel_inference as bbbc021_ds,  # noqa: E402
    )
    from my_conf.dataset.biotine_png_128_inference import dataset as biotine_ds  # noqa: E402
    from my_conf.dataset.diabetic_retinopathy_inference import (
        diabetic_retinopathy_inference as diabetic_retinopathy_ds,  # noqa: E402
    )
    from my_conf.dataset.NASH_fibrosis_inference import dataset as NASH_fibrosis_ds  # noqa: E402
    from my_conf.dataset.NASH_steatosis_inference import dataset as NASH_steatosis_ds  # noqa: E402

    datasets: list[DataSet] = [NASH_steatosis_ds, diabetic_retinopathy_ds, biotine_ds, bbbc021_ds, NASH_fibrosis_ds]
    device = "cuda:0"
    model_name = "facebook/dinov2-with-registers-giant"
    batch_size = 1024
    use_model_preprocessor = False
    recompute_encodings: bool | Literal["no-overwrite"] = "no-overwrite"
    save_times: Literal["no-overwrite"] | Literal["overwrite"] | Literal["ask-before-overwrite"] | Literal["no"] = (
        "overwrite"
    )
    seed = random.randint(0, 2**32 - 1)
    frac_range_delta = 0.3  # fraction of the true time range to use for t_min, t_max parametrization

    ###################################################################################################################
    ##################################################### Launch ######################################################
    ###################################################################################################################
    base_save_dir = Path("ordering_datasets")
    encoding_scheme = model_name.replace("/", "_") + (
        "_model_preproc" if use_model_preprocessor else "_dataset_preproc"
    )
    this_run_save_path = base_save_dir / encoding_scheme
    this_run_save_path.mkdir(parents=True, exist_ok=True)
    print("\n=> Parameters:")
    params = {
        "Device": device,
        "Model": model_name,
        "Datasets": [ds.name for ds in datasets],
        "Batch size": batch_size,
        "Use model preprocessor": use_model_preprocessor,
        "Recomputing encodings": recompute_encodings,
        "Saving times": save_times,
        "Seed": seed,
        "Delta used to extend the spline": frac_range_delta,
        "Base save dir": base_save_dir,
        "Run save path": this_run_save_path,
    }
    label_width = max(len(param) for param in params)
    for param, value in params.items():
        label_str = f"{param}:"
        dots = "_" * (label_width - len(label_str)) + ""
        print(f"    {label_str}{dots} {value}")
    print()
    inpt = input("=> Continue? (y/[]) ")
    if inpt != "y":
        print("Exiting...")
        sys.exit(0)

    # ruff: noqa: E402
    print("Loading imports... ", end="", flush=True)
    # Imports
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns
    import torch
    import umap
    from matplotlib.lines import Line2D
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
    print("done", flush=True)

    # Process each dataset iteratively
    for ds in datasets:
        print(f"\n{'#' * 60}")
        print(f"Processing dataset: {ds.name}")
        print(f"\n{'#' * 60}")

        this_run_save_path_this_ds = this_run_save_path / ds.name
        this_run_save_path_this_ds.mkdir(parents=True, exist_ok=True)

        # Save seed for reproducibility
        seed_file_path = this_run_save_path_this_ds / "seed.txt"
        with open(seed_file_path, "w") as f:
            f.write(str(seed))
        print(f"Saved seed to {seed_file_path}")
        # common rng for this run
        rng = np.random.default_rng(seed=seed)

        ###################################################################################################################
        ################################################ Compute encodings ################################################
        ###################################################################################################################
        encodings_filename = f"{encoding_scheme}_encodings.parquet"
        load_existing_encodings = False
        df: pd.DataFrame = pd.DataFrame()  # pylance is idiotic
        if recompute_encodings is not False:
            df_or_None = save_encodings(
                device,
                model_name,
                ds,
                use_model_preprocessor,
                this_run_save_path_this_ds,
                encodings_filename,
                batch_size,
                recompute_encodings,
            )
            if df_or_None is None:
                load_existing_encodings = True
            else:
                df = df_or_None
        else:
            load_existing_encodings = True

        if load_existing_encodings:
            print(f"\n=> Reusing existing encodings at {this_run_save_path_this_ds / encodings_filename}")
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
        print(f"cls_tokens.shape: {cls_tokens.shape}, len(labels): {len(labels)}, len(file_paths): {len(file_paths)}")

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
        print(f"\n=> Using sorted unique labels: {sorted_unique_labels}")
        sorted_unique_label_times = get_evenly_spaced_timesteps(len(sorted_unique_labels))
        print(f"Sorted unique label times: {sorted_unique_label_times}")

        ### UMAP
        print("\n=> UMAP")
        # 2D
        umap_2d_reducer = umap.UMAP(random_state=seed)
        umap_2d_embeddings: np.ndarray = umap_2d_reducer.fit_transform(cls_tokens)  # pyright: ignore[reportAssignmentType]
        plot_2D_embeddings(
            sorted_unique_labels, labels, umap_2d_embeddings, model_name, ds, rng, "UMAP", this_run_save_path_this_ds
        )
        # 3D
        umap_3d_reducer = umap.UMAP(random_state=seed, n_components=3)
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
            f"Seed={seed}",
        )

        ### PCA
        print("\n=> PCA")
        pca = PCA(random_state=seed)
        pca_embeddings = pca.fit_transform(cls_tokens)
        pca_explained_variance = pca.explained_variance_ratio_
        # 2D
        plot_2D_embeddings(
            sorted_unique_labels,
            labels,
            pca_embeddings,
            model_name,
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
            f"Seed={seed} | Total explained variance: {np.sum(pca_explained_variance[:3]) * 100:.1f}%",
            (
                f"PCA 1 ({pca_explained_variance[0] * 100:.1f}% of explained variance)",
                f"PCA 2 ({pca_explained_variance[1] * 100:.1f}% of explained variance)",
                f"PCA 3 ({pca_explained_variance[2] * 100:.1f}% of explained variance)",
            ),
        )

        # LDA
        print("\n=> LDA")
        lda = LinearDiscriminantAnalysis()
        lda_embeddings = lda.fit_transform(cls_tokens, labels)
        lda_explained_variance = lda.explained_variance_ratio_
        ## 2D
        plot_2D_embeddings(
            sorted_unique_labels,
            labels,
            lda_embeddings,
            model_name,
            ds,
            rng,
            "LDA",
            this_run_save_path_this_ds,
            f"Total explained variance: {np.sum(lda_explained_variance[:2]) * 100:.1f}%",
            (
                f"LDA 1 ({lda_explained_variance[0] * 100:.1f}% of explained variance)",
                f"LDA 2 ({lda_explained_variance[1] * 100:.1f}% of explained variance)",
            ),
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
        #     f"LDA projection of CLS tokens of {model_name} on {ds.name}\n"
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
        )

        ###################################################################################################################
        ############################################# Derive continuous time ##############################################
        ###################################################################################################################
        print("\n=> Deriving continuous time predictions from LDA embeddings")

        ### From pure proba (bad)
        print("\n-> from pure probabilities")
        proba = lda.predict_proba(cls_tokens)
        continuous_time_predictions = proba @ lda.classes_
        print(
            f"Computed continuous time predictions from LDA probabilities, shape: {continuous_time_predictions.shape}, excerpt: {continuous_time_predictions[:5]}"
        )
        plot_histograms_continuous_time_preds(
            this_run_save_path_this_ds, continuous_time_predictions, labels, "proba", True
        )
        plot_boxplots_continuous_time_preds(this_run_save_path_this_ds, labels, continuous_time_predictions, "proba")

        ### From decision function
        print("\n-> from decision function")
        # ie signed distance to the hyperplane
        scores = lda.decision_function(cls_tokens)
        continuous_time_predictions = scores @ lda.classes_
        print(
            f"Computed continuous time predictions from LDA decision function, shape: {continuous_time_predictions.shape}, excerpt: {continuous_time_predictions[:5]}"
        )
        continuous_time_predictions -= continuous_time_predictions.min()
        continuous_time_predictions /= continuous_time_predictions.max()
        print("Min-max normalized continuous time predictions to [0, 1]")
        plot_histograms_continuous_time_preds(
            this_run_save_path_this_ds, continuous_time_predictions, labels, "decision_func"
        )
        plot_boxplots_continuous_time_preds(
            this_run_save_path_this_ds, labels, continuous_time_predictions, "decision_func", len(df) <= 10_000
        )

        ### From projection on spline going through class centroids
        print("\n-> from projection on spline going through class centroids")
        print(f"Computing spline through class centroids of shape {lda.means_.shape}")  # pyright: ignore[reportAttributeAccessIssue]
        # Compute spline
        # (beware that embeddings must be that of the LDA here!)
        print(
            f"Fitting interpolating spline through class centroids of shape {lda.means_.shape} for labels: {sorted_unique_labels} at times {sorted_unique_label_times}"  # pyright: ignore[reportAttributeAccessIssue]
        )
        spline = make_interp_spline(sorted_unique_label_times, lda.means_)
        # Evaluate spline
        assert (sorted_unique_label_times[0], sorted_unique_label_times[-1]) == (0, 1)
        t_min, t_max = -frac_range_delta, 1 + frac_range_delta
        print(
            f"Using {frac_range_delta} of the 0-1 time range for t_min, t_max = {t_min}, {t_max} parametrization of the spline"
        )
        times_to_eval_spline = np.linspace(t_min, t_max, 1000)
        spline_values = spline(times_to_eval_spline)  # these are in DINO's embedding space
        # Project on spline # TODO: try projecting on the spline in the 3D LDA space as it is this space that is structured around time!
        continuous_time_predictions = project_to_time(cls_tokens, spline, t_min, t_max)
        print(
            f"Computed continuous time predictions from spline projection, shape: {continuous_time_predictions.shape}, excerpt: {continuous_time_predictions[:5]}"
        )
        # plot spline and projections along with base embeddings
        random_idx_to_plot_projections = rng.choice(len(cls_tokens), size=50, replace=False)
        projection_pairs = (
            cls_tokens[random_idx_to_plot_projections],
            spline(continuous_time_predictions[random_idx_to_plot_projections]),
        )
        # plot 3D embeddings with spline and projections for LDA, UMAP and PCA viz
        plot_3D_embeddings(
            labels,
            sorted_unique_labels,
            lda_embeddings,
            lda,
            rng,
            this_run_save_path_this_ds,
            ds,
            "LDA_splines",
            encoding_scheme,
            f"Total explained variance: {np.sum(lda_explained_variance[:3]) * 100:.1f}% | frac_range_delta={frac_range_delta}",
            (
                f"LDA 1 ({lda_explained_variance[0] * 100:.1f}% of explained variance)",
                f"LDA 2 ({lda_explained_variance[1] * 100:.1f}% of explained variance)",
                f"LDA 3 ({lda_explained_variance[2] * 100:.1f}% of explained variance)",
            ),
            spline_values,
            projection_pairs,
        )
        plot_3D_embeddings(
            labels,
            sorted_unique_labels,
            umap_3d_embeddings,
            umap_3d_reducer,
            rng,
            this_run_save_path_this_ds,
            ds,
            "UMAP_splines",
            encoding_scheme,
            f"Seed={seed} | frac_range_delta={frac_range_delta}",
            spline_raw_values=spline_values,
            projection_pairs=projection_pairs,
        )
        plot_3D_embeddings(
            labels,
            sorted_unique_labels,
            pca_embeddings,
            pca,
            rng,
            this_run_save_path_this_ds,
            ds,
            "PCA_splines",
            encoding_scheme,
            f"Seed={seed} | frac_range_delta={frac_range_delta}",
            spline_raw_values=spline_values,
            projection_pairs=projection_pairs,
        )
        # Plot histograms and boxplots of continuous time predictions
        plot_histograms_continuous_time_preds(
            this_run_save_path_this_ds, continuous_time_predictions, labels, "spline_projection"
        )
        plot_boxplots_continuous_time_preds(
            this_run_save_path_this_ds, labels, continuous_time_predictions, "spline_projection", len(df) <= 10_000
        )
        # plot distances to true labels
        plot_histograms_distances_to_true_labels(
            sorted_unique_label_times,
            sorted_unique_labels,
            continuous_time_predictions,
            ds.name,
            this_run_save_path_this_ds,
            labels,
        )
        # min-max-normalize to [0,1] # TODO: 5-95 percentile?
        continuous_time_predictions -= continuous_time_predictions.min()
        continuous_time_predictions /= continuous_time_predictions.max()
        plot_histograms_continuous_time_preds(
            this_run_save_path_this_ds, continuous_time_predictions, labels, "spline_projection_min_max_norm"
        )

        ### Save new continuous time label
        new_ds_file_save_path = (
            Path(ds.path).parent / f"{ds.name}__continuous_time_predictions__{encoding_scheme}.parquet"
        )
        print(f"Original dataset path:                        {ds.path}")
        print(f"Saving continuous time predictions dataset to {new_ds_file_save_path}")

        data = []
        for sample_file_path, sample_continuous_time in zip(file_paths, continuous_time_predictions, strict=True):
            data.append({"time": sample_continuous_time, "file_path": sample_file_path})
        df = pd.DataFrame(data)

        if save_times != "no":
            write_times = True
            if new_ds_file_save_path.exists():
                if save_times == "no-overwrite":
                    print(f"File {new_ds_file_save_path} already exists, skipping saving")
                    write_times = False
                elif save_times == "overwrite":
                    print(f"File {new_ds_file_save_path} already exists, overwriting")
                    write_times = True
                elif save_times == "ask-before-overwrite":
                    inpt = input(f"Delete existing file at {new_ds_file_save_path}? (y/n)")
                    if inpt == "y":
                        write_times = True
                    else:
                        print("Refusing to continue")
                        write_times = False
                else:
                    raise ValueError(f"Unknown save_times value: {save_times}")
            if write_times:
                df.to_parquet(new_ds_file_save_path, index=False)
                print(f"Saved continuous time predictions dataset to {new_ds_file_save_path}")
        else:
            print("No times saving")
