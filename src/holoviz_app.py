import numpy as np
import pandas as pd
import panel as pn

import holoviews as hv
import hvplot.pandas  # noqa
from streamz import Stream
import hvplot.streamz  # noqa
import param

from streamz.dataframe import Random

pn.extension(template="fast")
pn.extension("tabulator")

pn.extension("vega")

record_button = pn.widgets.Button(name="Record", button_type="primary")


text = pn.widgets.TextInput(value="Ready")


recorded_list = ["ja", "nein"]


def b(event):
    if record_button.button_type == "danger":
        record_button.button_type = "primary"
        recorded_list.append(text.value)
        print(recorded_list)
    else:
        record_button.button_type = "danger"


df = pd.read_csv("https://datasets.holoviz.org/penguins/v1/penguins.csv")

slider = pn.widgets.IntSlider(value=5, start=1, end=10, name="page_size")


record_button.on_click(b)
pn.Row(record_button, text)


def increment(x):
    return x + 1


source = Stream()

streamz_pane = pn.pane.Streamz(source.map(increment), always_watch=True)


def emit():
    df.emit(
        pd.DataFrame(
            {"y": [np.random.randn()]}, index=pd.DatetimeIndex([datetime.now()])
        )
    )


import altair as alt
from streamz.dataframe import DataFrame as sDataFrame
from datetime import datetime

df = sDataFrame(example=pd.DataFrame({"y": []}, index=pd.DatetimeIndex([])))


def line_plot(data):
    return (
        alt.Chart(pd.concat(data).reset_index())
        .mark_line()
        .encode(
            x="index",
            y="y",
        )
    )


altair_stream = df.cumsum().stream.sliding_window(50).map(line_plot)

altair_pane = pn.pane.Streamz(altair_stream, height=350, always_watch=True)

pn.state.add_periodic_callback(emit, period=100)

art = pn.Column(
    record_button,
    text,
    "".join(recorded_list),
    slider,
    altair_pane,
    sizing_mode="stretch_width",
).servable(area="sidebar")
