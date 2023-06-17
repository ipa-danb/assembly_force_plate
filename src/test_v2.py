from streamz import Stream
import pandas as pd
from streamz.dataframe import DataFrame
import panel as pn
import rospy
from geometry_msgs.msg import WrenchStamped
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
source3 = Stream(asynchronous=False)

source_df = DataFrame(example=convert_force_to_df())

source_df.stream.sink(print)


c = source1.zip(source2, source3)
ll3 = []

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
    if count % 2000 == 0:
        print(2000 / (rospy.Time.now() - t).to_sec())
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

# sdf = DataFrame(
#     c.filter(Counter(500).check).map(pd.DataFrame), example=convert_force_to_df()
# )


# sdf.sink(print)

# abra = c.filter(Counter(1000).check).sink(lambda x: print(pd.DataFrame(x)))


# def box_plots_dynamic(data_stream, column_name):
#     return hv.DynamicMap(
#         partial(hv.Curve, kdims=["ts"], vdims=column_name, label=column_name),
#         streams=[data_stream],
#     ).opts(ylim=(-20, 20))


def get_plot(data):
    data_selected = data.loc[data["name"] == "t1"][["ts", "fx", "fy", "fz"]]
    data_selected.reset_index()
    print(data_selected)
    return hv.Curve(data_selected, "ts", "fz", label="t1", width=800)


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
        save_pd = pd.DataFrame(itertools.chain.from_iterable(ll3))
        # print(save_pd)
        df_panel.object = save_pd
        source_df.emit(save_pd)
        ll3.clear()
        record_flag = False


button = pn.widgets.Button(name="start stream", button_type="primary")
button.on_click(on_click)


dial_x = pn.indicators.Dial(
    name="fx", value=10, bounds=(-20, 20), format="{value:.2f} N"
)
dial_y = pn.indicators.Dial(
    name="fy", value=10, bounds=(-20, 20), format="{value:.2f} N"
)
dial_z = pn.indicators.Dial(
    name="fz", value=10, bounds=(-20, 20), format="{value:.2f} N"
)

streamz_pane = pn.panel(df_panel)

plot_stream = source_df.stream.map(get_plot)
plot_panel = pn.pane.Streamz(plot_stream, example=hv.Curve(convert_force_to_df()))


def indicator_callback(df, dial, element_name):
    for element in zip(dial, element_name):
        element[0].value = df[0][element[1]]


dials = ([dial_x, dial_y, dial_z], ["fx", "fy", "fz"])

c.filter(Counter(100).check).sink(
    partial(indicator_callback, dial=dials[0], element_name=dials[1])
)


layout = pn.Row(
    pn.Column(button, streamz_pane),
    pn.Column(pn.Row(dial_x, dial_y, dial_z), plot_panel),
)

pn.serve(layout)

rospy.sleep(0.5)

rospy.Subscriber(
    "wrench", WrenchStamped, partial(callback, source=source1, name="t1"), queue_size=10
)
rospy.Subscriber(
    "wrench_test",
    WrenchStamped,
    partial(callback, source=source2, name="t2"),
    queue_size=10,
)
rospy.Subscriber(
    "wrench", WrenchStamped, partial(callback, source=source3, name="t3"), queue_size=10
)


rospy.spin()
