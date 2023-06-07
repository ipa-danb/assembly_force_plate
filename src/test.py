from functools import partial

import numpy as np
import panel as pn
import pandas as pd

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import WrenchStamped
from streamz import Stream

from functools import partial

pn.extension("vega")

rospy.init_node("test")


source = Stream()
L = source.sink_to_list()

df = pd.DataFrame({"x": [0.0], "y": [0.0]})
sdf = ColumnDataSource(df)

streamz_pane = pn.pane.Streamz(source, always_watch=False)

# source = ColumnDataSource({"x": np.arange(10), "y": np.arange(10)})


def panel_app():
    # p.line(x="x", y="y", source=source)
    return pn.Column(pn.pane.Streamz(source, always_watch=False))


plot = figure(
    height=800,
    width=800,
    title="plot",
    tools="crosshair,pan,reset,save,wheel_zoom",
    x_range=[0, 4 * np.pi],
    y_range=[-2.5, 2.5],
)

plot.line("x", "y", source=sdf, line_width=3, line_alpha=0.6)

df_widget = pn.widgets.DataFrame(df, name="DataFrame")
bokeh_app = pn.pane.Bokeh(plot)


def test_callback(msg):
    force = msg.wrench.force
    # source.emit(pd.DataFrame({"x": [force.x], "y": [force.y]}))
    # global df
    # df = pd.concat([df, pd.DataFrame({"x": [force.x], "y": [force.y]})])
    sdf.stream(pd.DataFrame({"x": [force.x], "y": [force.y]}))
    # df_widget.stream(df)


rospy.Subscriber("wrench", WrenchStamped, test_callback)


pn.serve(bokeh_app)
