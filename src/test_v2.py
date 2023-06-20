from streamz import Stream
import pandas as pd
from streamz.dataframe import DataFrame
import panel as pn
import rospy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Empty
import holoviews as hv
from functools import partial

hv.extension("bokeh")
pn.extension()

rospy.init_node(name="test", anonymous=True)

list_of_wrench_topics = ["wrench"]


def convert_force_to_dict(msg=None, name=""):
    rev = {}
    rev["ts"] = pd.Timestamp(msg.header.stamp.to_sec(), unit="s")
    rev["fx"] = msg.wrench.force.x  # /1000000]
    rev["fy"] = msg.wrench.force.y  # /1000000]
    rev["fz"] = msg.wrench.force.z  # /1000000]
    rev["tx"] = msg.wrench.torque.x  # /1000000]
    rev["ty"] = msg.wrench.torque.y  # /1000000]
    rev["tz"] = msg.wrench.torque.z  # /1000000]
    rev["name"] = name
    return rev


def convert_force_to_df():
    rev = {}
    rev["ts"] = [pd.Timestamp(1686923414098406572, unit="s")]
    rev["fx"] = [0]
    rev["fy"] = [0]
    rev["fz"] = [0]
    rev["tx"] = [0]
    rev["ty"] = [0]
    rev["tz"] = [0]
    rev["name"] = [""]
    reval = pd.DataFrame(rev)  # , index=pd.Timestamp(1686923414098406572, unit="s"))
    # reval.set_index("ts", inplace=True)
    return reval


source1 = Stream(asynchronous=False)
source2 = Stream(asynchronous=False)
source_df = Stream(asynchronous=False)


multi_select = pn.widgets.MultiSelect(
    name="MultiSelect",
    value=["Apple", "Pear"],
    options=["Apple", "Banana", "Pear", "Strawberry"],
    size=8,
)

# source_df = DataFrame(source3, example=convert_force_to_df())

# source_df.sink(print)


def add_option(op, multiselect):
    multiselect.options.append(f"Recording {len(multiselect.options)}")


c = source1.zip(source2)
ll3 = []
recordings = []

# recs = source_df.sink(recordings.append)
# source_df.sink(partial(add_option, multiselect=multi_select))

tt = c.sink(ll3.append)


c.disconnect(tt)
record_flag = False

count = 0
t = rospy.Time.now()


def callback(msg, source, name):
    cf = partial(convert_force_to_dict, name=name)
    source.emit(cf(msg))


def test_callback(msg):
    global count, t
    count += 1
    if count % 300 == 0:
        print(300 / (rospy.Time.now() - t).to_sec())
        print(ll3[-1:])
        print(len(ll3))
        t = rospy.Time.now()


class Counter:
    def __init__(self, trueval):
        self.counter = 0
        self.trueval = trueval

    def check(self, *args):
        self.counter += 1
        if self.counter == self.trueval:
            self.counter = 0
            return True
        else:
            return False


c.sink(test_callback)


def get_plot(data):
    print("start getplot")
    names = data["name"].unique()
    t0 = data.loc[data["name"] == "t0"]["fz"].reset_index()
    t1 = data.loc[data["name"] == "t1"]["fz"].reset_index()
    ratio = (t0["fz"] + 10).div(t1["fz"] + t0["fz"] + 10)
    ratio.loc[ratio > 1.0] = 1.0
    ratio.loc[ratio < -1.0] = -1.0
    print(t0, t1, ratio)
    rr = []
    for name in names:
        data_selected = data.loc[data["name"] == name][["ts", "fx", "fy", "fz"]]
        data_selected.reset_index()
        rr.append(
            hv.Overlay(
                [
                    hv.Curve(data_selected, "ts", f, label=name)
                    for f in ["fx", "fy", "fz"]
                ]
            )
        )
    rr.append(hv.Curve(ratio).opts(shared_axes=False))
    return hv.Layout(rr).cols(3)


save_pd = None


import itertools

df_panel = pn.pane.DataFrame(convert_force_to_df(), width=400)


def on_click(event):
    global record_flag, c, tt, save_pd, ll3, sdf, plot_panel
    print("clicked")
    print("record_flag: ", record_flag)
    if not record_flag:
        print("try to connect")
        c.connect(tt)
        record_flag = True
    else:
        c.disconnect(tt)
        print("convert")
        save_pd = None
        save_pd = pd.DataFrame(itertools.chain.from_iterable(ll3))
        df_panel.object = save_pd
        print(save_pd)
        source_df.emit(save_pd, asynchronous=True)
        print("finish on click")
        ll3.clear()
        record_flag = False


class ResetCaller:
    def __init__(self, topic_names):
        self.s = [
            rospy.Publisher(
                topic_name,
                Empty,
                queue_size=10,
            )
            for topic_name in topic_names
        ]

    def call_reset(self, event):
        for element in self.s:
            element.publish()


reset_topics = [
    "/sensor_01/ati_force_sensor/set_sw_bias",
    "/sensor_02/ati_force_sensor/set_sw_bias",
]


reset_button = pn.widgets.Button(name="reset FT", button_type="primary")

rscall = ResetCaller(reset_topics)
# reset_button.on_click(rscall.call_reset)
reset_button.on_click(print)


button = pn.widgets.Button(name="start stream", button_type="primary")
# button.on_click(on_click)
button.on_click(print)


dial_x_1 = pn.indicators.Dial(
    name="fx",
    value=10,
    bounds=(-20, 20),
    format="{value:.2f} N",
    width=200,
)
dial_y_1 = pn.indicators.Dial(
    name="fy",
    value=10,
    bounds=(-20, 20),
    format="{value:.2f} N",
    width=200,
)
dial_z_1 = pn.indicators.Dial(
    name="fz",
    value=10,
    bounds=(-20, 20),
    format="{value:.2f} N",
    width=200,
)

dial_x_2 = pn.indicators.Dial(
    name="fx",
    value=10,
    bounds=(-20, 20),
    format="{value:.2f} N",
    width=200,
)
dial_y_2 = pn.indicators.Dial(
    name="fy",
    value=10,
    bounds=(-20, 20),
    format="{value:.2f} N",
    width=200,
)
dial_z_2 = pn.indicators.Dial(
    name="fz",
    value=10,
    bounds=(-20, 20),
    format="{value:.2f} N",
    width=200,
)


streamz_pane = pn.panel(df_panel)

plot_stream = source_df.map(get_plot)
plot_panel = pn.pane.Streamz(plot_stream)


def bound_value(low, high, value):
    return max(low, min(high, value))


def indicator_callback(df, dials, element_names):
    for i, (dial, element_name) in enumerate(zip(dials, element_names)):
        for element in zip(dial, element_name):
            element[0].value = bound_value(-20.0, 20.0, df[i][element[1]])


dials_1 = ([dial_x_1, dial_y_1, dial_z_1], ["fx", "fy", "fz"])
dials_2 = ([dial_x_2, dial_y_2, dial_z_2], ["fx", "fy", "fz"])


kr = c.filter(Counter(10).check)
kr.sink(
    partial(
        indicator_callback,
        dials=[dials_1[0], dials_2[0]],
        element_names=[dials_1[1], dials_2[1]],
    )
)


def force_threshold(df, dim, thresh):
    ff = sum([abs(element[dim]) for element in df])
    if ff > thresh:
        return True
    else:
        return False


def calc_ratio(df, dim):
    reval = abs(df[0][dim]) / abs(df[0][dim] + df[1][dim])
    return bound_value(-0.2, 1.2, reval)


blub = kr.filter(partial(force_threshold, dim="fz", thresh=0.5))


indicator = pn.indicators.LinearGauge(
    name="Position",
    value=15.0,
    bounds=(-1.0, 26.0),
    format="{value:.2f} cm",
    horizontal=True,
    height=600,
)


def set_dial_value(value, indicator):
    print(f"set_value: {value}")
    indicator.value = float(value)
    print(f"inicator value: {indicator.value}")


def analyze_print(element):
    print(f"{type(element)}: {element}")


wellp = (
    blub.map(partial(calc_ratio, dim="fz"))
    .map(lambda x: x * 24.5)
    .map(partial(bound_value, 0.0, 25.0))
    .sink(partial(set_dial_value, indicator=indicator))
)
wellp.sink(analyze_print)


layout = pn.Row(
    pn.Column(pn.Row(button, reset_button), streamz_pane),
    pn.Column(
        indicator,
        pn.Row(dial_x_1, dial_y_1, dial_z_1),
        pn.Row(dial_x_2, dial_y_2, dial_z_2),
        plot_panel,
    ),
)

pn.serve(layout)

rospy.sleep(2.5)

wrenches = [
    "/sensor_01/ati_force_sensor/ati_force_torque",
    "/sensor_02/ati_force_sensor/ati_force_torque",
]

sources = [source1, source2]


subscribers = [
    rospy.Subscriber(
        topic_name,
        WrenchStamped,
        partial(callback, source=s, name=f"t{its}"),
        queue_size=10,
    )
    for its, (topic_name, s) in enumerate(zip(wrenches, sources))
]

rscall.call_reset(None)


rospy.spin()
