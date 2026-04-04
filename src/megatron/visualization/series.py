import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from megatron.transformers.series import (
    ChangePointDetector,
    PlateauDetector,
    OutlierDetector,
)

import megatron.config as config


def seriesPlot(
    data: pd.DataFrame,
    demand="",
    X_exog=None,
    exog_column=None,
    w=config.SEASONAL_PERIOD,  # type: ignore
    n_series=1,
    title="",
    pld=False,
    pd_value=np.nan,
    cpd=False,
    od=False,
    seed=config.SEED,  # type: ignore
    line_width=1.5,
) -> None:

    if data.index.nlevels > 1:
        np.random.seed(seed)
        index = np.random.choice(
            data.droplevel(-1).index.unique(), size=n_series, replace=False
        )
    else:
        data = (
            data.assign(index=data.columns[0])
            .set_index("index", append=True)
            .reorder_levels(["index", "date"])
        )
        index = data.droplevel(-1).index.unique()

    n_rows, gap = n_series // 2 if n_series > 1 else 1, 50
    height = max(n_rows * 250 + (n_rows - 1) * gap, config.FIG_HEIGHT)  # type: ignore
    v_space = gap / height if n_rows > 1 else 0
    plt = make_subplots(
        rows=n_rows,
        cols=2 if n_series > 1 else 1,
        subplot_titles=[str(x) for x in index],
        horizontal_spacing=0.05,
        vertical_spacing=v_space,
    )

    for i, instance in enumerate(index):
        temp = data.loc[instance]

        plt.add_trace(
            go.Scatter(
                x=temp.index,
                y=temp.values.flatten(),
                line={"color": config.COLOR, "width": line_width},  # type: ignore
            ),
            row=i // 2 + 1,
            col=i % 2 + 1,
        )

        if pld:
            result = PlateauDetector(w=w, value=pd_value).fit_transform(temp)
            if result.size:  # type: ignore
                for plateau in result:  # type: ignore
                    plt.add_vrect(
                        x0=plateau[0],
                        x1=plateau[-1],
                        fillcolor="#E2A20C",
                        opacity=0.2,
                        line_width=0.1,
                        row=i // 2 + 1,  # type: ignore
                        col=i % 2 + 1,  # type: ignore
                    )

        if cpd:
            result = ChangePointDetector(w=w).fit_transform(temp)
            plt.add_vrect(
                x0=result,  # type: ignore
                x1=temp.index[-1],
                fillcolor="#3C9B1F",
                opacity=0.2,
                line_width=0.1,
                row=i // 2 + 1,  # type: ignore
                col=i % 2 + 1,  # type: ignore
            )

        if od:
            if X_exog is not None:
                temp = temp.join(X_exog.loc[instance], how="inner")
            result = OutlierDetector(
                demand=demand, exog_column=exog_column
            ).fit_transform(temp)
            plt.add_scatter(
                x=result.index,  # type: ignore
                y=result.values.flatten(),  # type: ignore
                mode="markers",
                marker={"size": 5, "color": "#CC2411"},
                row=i // 2 + 1,
                col=i % 2 + 1,
            )

    else:
        plt.update_layout(
            width=config.FIG_WIDTH,  # type: ignore
            height=height,  # type: ignore
            showlegend=False,
            title={"text": title, "x": 0.5},
            margin=config.MARGIN,  # type: ignore
        )
        plt.show()


def clusteredSeriesPlot(
    data: pd.DataFrame, title="", line_width=1.5
):
    index, value = data.index.names, data.columns[0]
    clusters = data.index.get_level_values(0).unique()
    n_rows = int(np.ceil(len(clusters) / 2))

    n_rows, gap = len(clusters) // 2 if len(clusters) > 1 else 1, 50
    height = max(n_rows * 250 + (n_rows - 1) * gap, config.FIG_HEIGHT)  # type: ignore
    v_space = gap / height if n_rows > 1 else 0
    plt = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=[f"Cluster {x}" for x in clusters],
        horizontal_spacing=0.05,
        vertical_spacing=v_space,
    )

    for i, cluster in enumerate(clusters):
        temp = data.loc[cluster].reset_index()

        for _, temp_item in temp.groupby(index[1]):
            plt.add_trace(
                go.Scatter(
                    x=temp_item[index[-1]],
                    y=temp_item[value],
                    line={"width": line_width},
                ),
                row=i // 2 + 1,
                col=i % 2 + 1,
            )
    else:
        plt.update_layout(
            width=config.FIG_WIDTH,  # type: ignore
            height=height,  # type: ignore
            showlegend=False,
            title={"text": title, "x": 0.5},
            margin=config.MARGIN,  # type: ignore
        )
        plt.show()


def forecastedSeriesPlot(
    data: pd.DataFrame,
    group="group",
    n_series=1,
    title="",
    seed=config.SEED,  # type: ignore
    line_width=1.5,
) -> None:

    value = data.columns[0]
    np.random.seed(seed)
    index = np.random.choice(
        data.droplevel(-1).index.unique(), size=n_series, replace=False
    )

    n_rows, gap = n_series // 2 if n_series > 1 else 1, 50
    height = max(n_rows * 250 + (n_rows - 1) * gap, config.FIG_HEIGHT)  # type: ignore
    v_space = gap / height if n_rows > 1 else 0
    plt = make_subplots(
        rows=n_rows,
        cols=2 if n_series > 1 else 1,
        subplot_titles=[str(x) for x in index],
        horizontal_spacing=0.05,
        vertical_spacing=v_space,
    )

    colors = {"train": "#636EFA", "forecast": "#EF553B"}

    for i, instance in enumerate(index):
        temp = data.loc[instance].reset_index()

        for g, temp_group in temp.groupby(group):
            plt.add_trace(
                go.Scatter(
                    x=temp_group[data.index.names[-1]],
                    y=temp_group[value],
                    name=g,
                    line={"width": line_width, "color": colors[g]},
                    mode="lines"
                ),
                row=i // 2 + 1,
                col=i % 2 + 1,
            )

    plt.update_layout(
        width=config.FIG_WIDTH,  # type: ignore
        height=height,  # type: ignore
        showlegend=False,
        title={"text": title, "x": 0.5},
        margin=config.MARGIN,  # type: ignore
    )
    plt.show()
