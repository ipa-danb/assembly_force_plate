from streamz import Stream
import pandas as pd
from streamz.dataframe import DataFrame
import panel as pn
import rospy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Empty
import holoviews as hv
from functools import partial
import numpy as np
from rich import print
from sklearn.mixture import BayesianGaussianMixture
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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


source1 = Stream(asynchronous=True, ensure_io_loop=True)
source2 = Stream(asynchronous=True, ensure_io_loop=True)
source_df = Stream(asynchronous=True, ensure_io_loop=True)


multi_select = pn.widgets.MultiSelect(
    name="Recordings",
    value=[],
    options=[],
    size=8,
)

# source_df = DataFrame(source3, example=convert_force_to_df())

# source_df.sink(print)


def save_to_csv(ll):
    for i, lr in enumerate(ll):
        print(f"save Recording {i}")
        lr.to_csv(f"Recording_{i}.csv")


def add_option(op, multiselect):
    print(multiselect.options)
    ak = multiselect.options.copy()
    ak.append(f"Recording {len(multiselect.options)}")
    multiselect.options = ak
    print(multiselect.options)


c = source1.zip(source2)
ll3 = []

rr = source_df.sink(partial(add_option, multiselect=multi_select))

recordings = source_df.sink_to_list()

tt = c.sink(ll3.append)


c.disconnect(tt)
record_flag = False

count = 0
t = rospy.Time.now()


def callback(msg, source, name):
    cf = partial(convert_force_to_dict, name=name)
    source.emit(cf(msg), asynchronous=False)


def test_callback(msg):
    global count, t
    count += 1
    if count % 3000 == 0:
        print(3000 / (rospy.Time.now() - t).to_sec())
        # print(ll3[-1:])
        # print(len(ll3))
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


# c.sink(test_callback)

str_pane = pn.pane.Str("Skills.", styles={"font-size": "12pt"})


def extract_data(frame):
    t0 = frame[frame["name"] == "t0"].reset_index()
    rec = t0[["fx", "fy", "fz"]]
    return rec.reset_index().to_numpy(), rec


def calculate_dpgmm(frame_list):
    dat_list = [extract_data(name)[0] for name in frame_list]
    dat_aligned = []
    x_n, y_n = align_two_arrays(dat_list[0], dat_list[1])
    dat_aligned.append(x_n)
    if len(dat_list) > 2:
        for el in dat_list[2:]:
            _, y_n = align_two_arrays(x_n, el)
            dat_aligned.append(y_n)
    dats = np.vstack(dat_aligned)
    # dfs = pd.concat([extract_data(name)[1] for name in frame_list])
    alder = BayesianGaussianMixture(
        n_components=10,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_distribution",
    ).fit(dats)
    selector = np.unique(alder.predict(dats))
    print(alder.means_.shape, alder.means_[selector].shape)
    print(alder.covariances_.shape, alder.covariances_[selector].shape)
    return alder.means_[selector], alder.covariances_[selector], dats


def print_stats(means, covariances, axes=[0, 3]):
    extracted_variances = np.diagonal(covariances, axis1=1, axis2=2)[:, axes]
    extracted_means = means[:, axes]
    force_skills, selection, means_sorted = extract_contact_establishers(
        means, covariances
    )
    suggestion_string = create_suggestions(means_sorted, selection)
    str_pane.object = "".join(suggestion_string)
    print(suggestion_string)
    for i, (m, c) in enumerate(zip(extracted_means, extracted_variances)):
        print(f"skill_{i}")
        print(
            f"extracted mean: {m[-1]:.2f}+/-{c[-1]:.2f} at {int(m[0]):d}+/-{int(c[0]):d}"
        )


def extract_contact_establishers(means, covariances, axes=[3]):
    ara = means[means[:, 0].argsort()]
    covs = covariances[means[:, 0].argsort()]
    print(ara)
    extracted_variances = np.diagonal(covs, axis1=1, axis2=2)[:, axes]
    extracted_means = ara[:, axes].flatten()
    print("extracted means")
    print(extracted_means)
    means_thresh = 2.0
    variances_thresh = 1.0
    print(np.abs(extracted_means) > means_thresh)
    print(ara[np.abs(extracted_means) > means_thresh])
    return (
        ara[np.abs(extracted_means) > means_thresh],
        [np.abs(extracted_means) > means_thresh],
        ara,
    )


def create_suggestions(ara, selection):
    contact_establish_skill = np.diff(selection, prepend=False)
    print_list = []
    print(contact_establish_skill, selection)
    for i, (m, contact_flag, force_skill) in enumerate(
        zip(ara, contact_establish_skill[0], selection[0])
    ):
        t0 = f"Skill {i}\n"
        if force_skill:
            if contact_flag:
                t1 = f"\t Skill_approach\n"
            else:
                t1 = f"\t Skill_apply_force\n"
            t2 = f"\t Forces: \t Fx:\t{m[1]:.1f}, Fy:\t{m[2]:.1f}, Fz:\t{m[3]:.1f}\n"
        else:
            t1 = "\t Preposition Skill\n"
            t2 = "\n"

        print(t0, t1, t2)
        tg = "\n".join([t0, t1, t2])
        print_list.append(tg)
    return print_list


def get_singular_plot(data, name, reset_index=False):
    xplot = "ts"
    data_copy = data.copy()
    if reset_index:
        data_copy["ts"] = data_copy["ts"] - data_copy["ts"][0]
        data_copy.reset_index()
    return [hv.Curve(data_copy, xplot, f, label=name) for f in ["fx", "fy", "fz"]]


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
        rr.append(hv.Overlay(get_singular_plot(data, name)))
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
        # print(save_pd)
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


dial_x_1 = pn.indicators.Dial(
    name="fx",
    value=10,
    bounds=(-20, 20),
    format="{value:.2f} N",
    width=200,
    default_color="red",
)
dial_y_1 = pn.indicators.Dial(
    name="fy",
    value=10,
    bounds=(-20, 20),
    format="{value:.2f} N",
    width=200,
    default_color="green",
)
dial_z_1 = pn.indicators.Dial(
    name="fz",
    value=10,
    bounds=(-20, 20),
    format="{value:.2f} N",
    width=200,
    default_color="blue",
)

dial_x_2 = pn.indicators.Dial(
    name="fx",
    value=10,
    bounds=(-20, 20),
    format="{value:.2f} N",
    width=200,
    default_color="red",
)
dial_y_2 = pn.indicators.Dial(
    name="fy",
    value=10,
    bounds=(-20, 20),
    format="{value:.2f} N",
    width=200,
    default_color="green",
)
dial_z_2 = pn.indicators.Dial(
    name="fz",
    value=10,
    bounds=(-20, 20),
    format="{value:.2f} N",
    width=200,
    default_color="blue",
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


kr = c.filter(Counter(150).check)
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


indicator = pn.indicators.LinearGauge(
    name="Position",
    value=15.0,
    bounds=(-1.0, 26.0),
    format="{value:.2f} cm",
    horizontal=True,
    height=600,
)


def set_dial_value(value, indicator):
    # print(f"set_value: {value}")
    indicator.value = float(value)
    # print(f"inicator value: {indicator.value}")


def analyze_print(element):
    print(f"{type(element)}: {element}")


def print_event(event):
    print("click!", flush=True)


def on_analyze(event, hv_panel):
    print(len(recordings))
    names = recordings[0]["name"].unique()
    print(len([get_singular_plot(data, name) for name in names for data in recordings]))
    ele = [get_singular_plot(data, "t0", reset_index=True) for data in recordings]
    print([el[2] for el in ele])
    hv_panel.object = hv.Overlay([el[2] for el in ele])
    print(event)
    print(hv_panel)


blub = kr.filter(partial(force_threshold, dim="fz", thresh=1.0))

wellp = (
    blub.map(partial(calc_ratio, dim="fz"))
    .map(lambda x: x * 24.5)
    .map(partial(bound_value, 0.0, 25.0))
    .sink(partial(set_dial_value, indicator=indicator))
)
wellp.sink(analyze_print)


reset_button = pn.widgets.Button(name="reset FT", button_type="primary")

rscall = ResetCaller(reset_topics)
reset_button.on_click(rscall.call_reset)
reset_button.on_click(print_event)


button = pn.widgets.Button(name="start stream", button_type="primary")
button.on_click(on_click)
button.on_click(print_event)


def analyzer(event, hv_panel):
    ms, cs, aligned_data = calculate_dpgmm(recordings)
    print("FZ Stats")
    print_stats(ms, cs, [0, 3])
    layout = hv.Overlay(
        [hv.Scatter(aligned_data[:, [0, 3]]), hv.Scatter(ms[:, [0, 3]])]
    )
    hv_panel.object = layout


analyze_button = pn.widgets.Button(
    name="Plot and analyze selected data", button_type="primary"
)
analyze_button.on_click(print_event)
analyze_panel = pn.panel(hv.Curve(None))
align_panel = pn.panel(hv.Curve(None), width=800, height=400)

analyze_button.on_click(partial(analyzer, hv_panel=align_panel))
analyze_button.on_click(partial(on_analyze, hv_panel=analyze_panel))


def save_list(event):
    save_to_csv(recordings)


def load_list(event):
    global recordings, multi_select
    import glob

    files = glob.glob("Recording*.csv")
    print(f"Loading {files}")
    recordings = [pd.read_csv(f, parse_dates=["ts"]) for f in files]
    multi_select.options = files


def align_two_arrays(x, y):
    distance, path = fastdtw(x[:, 1:], y[:, 1:], dist=euclidean)
    new_x_1 = np.array([p[0] for p in path])
    new_x_2 = np.array([p[1] for p in path])
    a = np.concatenate(
        (np.array([range(0, x[:, 1:][new_x_1].shape[0])]).T, x[:, 1:][new_x_1]), axis=1
    )
    b = np.concatenate(
        (np.array([range(0, y[:, 1:][new_x_2].shape[0])]).T, y[:, 1:][new_x_2]), axis=1
    )

    return a, b


save_button = pn.widgets.Button(name="Save data to csv", button_type="primary")
save_button.on_click(save_list)

load_button = pn.widgets.Button(name="Load data from files", button_type="primary")
load_button.on_click(load_list)


layout = pn.Row(
    pn.Column(
        pn.Row(button, reset_button),
        multi_select,
        pn.Row(analyze_button, save_button, load_button),
        streamz_pane,
    ),
    pn.Column(
        indicator,
        pn.Row(dial_x_1, dial_y_1, dial_z_1),
        pn.Row(dial_x_2, dial_y_2, dial_z_2),
        plot_panel,
        analyze_panel,
        align_panel,
    ),
    str_pane,
)


# rospy.sleep(2.5)

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

pn.serve(
    layout,
    threaded=True,
    show=True,
)

rospy.spin()
