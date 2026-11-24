"""Altair chart builders for the word-meaning lab (no Streamlit; unit-tested).

Word chart (page 1): each band (citizens, LLMs) is drawn as a lightly shaded footprint of uncertainty (between
LMF and UMF) PLUS solid outlines of both UMF and LMF, so the whole word stays visible: where all groups or models
agree, the LMF outline shows it, and the chart does not rely on colour alone. Draw order (bottom to top):
LLM fill, LLM outlines, model lines, citizen-group lines, citizen fill, citizen outlines. The citizen band is
drawn last, so the LLM fill can never hide the citizens' edges.
"""
from __future__ import annotations

import altair as alt
import pandas as pd

BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#8a8984'
MODEL_COLOURS = ['#1baf7a', '#eda100', '#e87ba4', '#6250d6']  # fixed order, one per model
CITIZEN_BAND, LLM_BAND = 'Citizens (band over 6 groups)', 'LLMs (band over models)'
FILL_OPACITY = .15
LAYER_ORDER = ['llm_fill', 'llm_outline', 'model_lines', 'group_lines', 'citizen_fill', 'citizen_outline']


def outlines(bands: pd.DataFrame) -> pd.DataFrame:
    """Long frame of the UMF and LMF outlines of each band: x, series, edge, mu."""
    up = bands.assign(edge='upper edge (UMF)', mu=bands.hi)
    low = bands.assign(edge='lower edge (LMF)', mu=bands.lo)
    return pd.concat([up, low], ignore_index=True)[['x', 'series', 'edge', 'mu']]


def word_chart(bands: pd.DataFrame, lines: pd.DataFrame, height: int = 360) -> alt.LayerChart:
    """Layered chart for one term; layer names follow LAYER_ORDER (layers without data are skipped)."""
    x = alt.X('x:Q', title='Probability (%)', scale=alt.Scale(domain=[0, 100]))
    y_title, y_scale = 'Membership (how well the number fits the word)', alt.Scale(domain=[0, 1.05])
    y = alt.Y('mu:Q', title=y_title, scale=y_scale)
    band_colour = alt.Color('series:N', title='Band', legend=alt.Legend(labelLimit=300),
                            scale=alt.Scale(domain=[CITIZEN_BAND, LLM_BAND], range=[BLUE, ORANGE]))
    edges = outlines(bands)
    tip_edge = ['series:N', 'edge:N', alt.Tooltip('x:Q', title='%'), alt.Tooltip('mu:Q', format='.2f', title='membership')]

    def fill(series, name):
        part = bands[bands.series == series]
        return alt.Chart(part).mark_area(opacity=FILL_OPACITY).encode(
            x=x, y=alt.Y('lo:Q', title=y_title, scale=y_scale), y2='hi:Q', color=band_colour,
            tooltip=['series:N', alt.Tooltip('x:Q', title='%'), alt.Tooltip('lo:Q', format='.2f', title='lower edge'),
                     alt.Tooltip('hi:Q', format='.2f', title='upper edge')]).properties(name=name)

    def edge(series, name):
        part = edges[edges.series == series]
        return alt.Chart(part).mark_line(strokeWidth=2).encode(
            x=x, y=y, color=band_colour, detail='edge:N', tooltip=tip_edge).properties(name=name)

    layers = [fill(LLM_BAND, 'llm_fill'), edge(LLM_BAND, 'llm_outline')]
    models = lines[lines.kind == 'model']
    if len(models):
        names = sorted(models.series.unique())
        layers.append(alt.Chart(models).mark_line(strokeWidth=1.5, strokeDash=[5, 3]).encode(
            x=x, y=y, stroke=alt.Stroke('series:N', title='LLM (dashed)',
                                        scale=alt.Scale(domain=names, range=MODEL_COLOURS[:len(names)])),
            tooltip=['series:N', alt.Tooltip('x:Q', title='%'), alt.Tooltip('mu:Q', format='.2f', title='membership')]
        ).properties(name='model_lines'))
    groups = lines[lines.kind == 'citizen group']
    if len(groups):
        layers.append(alt.Chart(groups).mark_line(strokeWidth=1, color=BLUE, opacity=.7, strokeDash=[2, 2]).encode(
            x=x, y=y, detail='series:N',
            tooltip=[alt.Tooltip('series:N', title='Citizen group'), alt.Tooltip('x:Q', title='%'),
                     alt.Tooltip('mu:Q', format='.2f', title='membership')]).properties(name='group_lines'))
    layers += [fill(CITIZEN_BAND, 'citizen_fill'), edge(CITIZEN_BAND, 'citizen_outline')]
    return alt.layer(*layers).resolve_scale(color='shared', stroke='independent').properties(height=height)


def layer_names(chart: alt.LayerChart) -> list[str]:
    return [layer.name for layer in chart.layer]
