import plotly.graph_objects as go


def apply_standard_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str = "Data",
    yaxis_title: str = "NDVI [0–1]",
) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=640,
        margin=dict(l=40, r=40, t=70, b=40),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    return fig


def figure_card(
    fig: go.Figure,
    title: str,
    intro: str = "",
    section_id: str = "fig",
    xaxis_title: str = "Data",
    yaxis_title: str = "NDVI [0–1]",
) -> str:
    intro_html = f"<p class='muted'>{intro}</p>" if intro else ""
    fig = apply_standard_layout(
        fig,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    return (
        f"<section class='card'>"
        f"<h1>{title}</h1>"
        f"{intro_html}"
        f"<div class='plot-wrap'><div id='{section_id}' class='plot-large'></div></div>"
        f"<script>"
        f"const spec_{section_id} = {fig.to_json()};"
        f"Plotly.newPlot('{section_id}', spec_{section_id}.data, spec_{section_id}.layout, {{responsive:true}});"
        f"</script>"
        f"</section>"
    )