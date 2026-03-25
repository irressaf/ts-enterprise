import numpy as np
import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from megatron.transformers.series import (
    ChangePointDetector,
    PlateauDetector,
    OutlierDetector,
)

import megatron.config as config

px.defaults.width, px.defaults.height = 900, 500  # type: ignore
margin, color = dict(l=30, r=30, t=50, b=30), "#1f77b4"


def seriesPlot(
    data: pd.DataFrame,
    w=config.SEASONAL_PERIOD,
    n_series=1,
    title="",
    pld=False,
    pd_value=np.nan,
    cpd=False,
    od=False,
    seed=42,
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

    plt = make_subplots(
        rows=n_series // 2 if n_series > 1 else 1,
        cols=2 if n_series > 1 else 1,
        subplot_titles=[str(x) for x in index],
        horizontal_spacing=0.05,
        vertical_spacing=0.075,
    )

    for i, instance in enumerate(index):
        temp = data.loc[instance]

        plt.add_trace(
            go.Scatter(
                x=temp.index,
                y=temp.values.flatten(),
                line={"color": color, "width": line_width},
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
            result = OutlierDetector().fit_transform(temp)  # type: ignore
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
            height=230 * n_series // 2 if n_series > 2 else 500,
            showlegend=False,
            title={"text": title, "x": 0.5},
            margin=margin,
        )
        plt.show()
