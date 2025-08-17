import numpy as np
import pandas as pd
from functools import partial
from rich import print
from threading import Thread
from collections import defaultdict
import glob

from streamz import Stream
from streamz.dataframe import DataFrame
from sklearn.mixture import BayesianGaussianMixture

import rospy
from std_msgs.msg import Empty
import rust_subscriber

import panel as pn
import holoviews as hv
from holoviews import dim

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


hv.extension("bokeh")
pn.extension()

debug = False


rospy.init_node(name="recording_gui", anonymous=True)


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


class ResetCaller:
    def __init__(self, topic_names):
        self.s = [
            rospy.Publisher(
                topic_name,
                Empty,
                queue_size=2,
            )
            for topic_name in topic_names
        ]

    def call_reset(self, event):
        for element in self.s:
            for i in range(0, 10):
                element.publish()


header = ["ts"]
header.extend([f"{f}{dim}" for f in "fm" for dim in "xyz"])


def add_option(op, multiselect):
    ak = multiselect.options.copy()
    ak.append(f"Recording {len(multiselect.options)}")
    multiselect.options = ak


def save_to_csv(ll):
    for i, lr in enumerate(ll):
        print(f"save Recording {i}")
        lr.to_csv(f"Recording_{i}.csv")


def extract_data(data_frame, name=1, measurement_names=["fx", "fy", "fz"]):
    """Extracts dataframe where the name is the same as arg name and selects only measurements

    Args:
        data_frame (DataFrame): complete Dataframe
        name (int, optional): name. Defaults to 1.
        measurement_names (list, optional): list of measurements to extract. Defaults to ["fx", "fy", "fz"].

    Returns:
        tuple(numpy.array, dataframe): returns the values as either numpy array or dataframe
    """
    print("extract_data")
    print(data_frame)
    print(name)
    sub_data_frame = data_frame[data_frame["name"] == name].reset_index()
    ret_frame = sub_data_frame[measurement_names]
    return ret_frame.reset_index().to_numpy(), ret_frame


def create_df(x):
    header = ["ts"]
    header.extend([f"{f}{dim}" for f in "fm" for dim in "xyz"])
    return pd.DataFrame([x], columns=header)


def create_dict(x, name=""):
    d = {t1: t2 for (t1, t2) in zip(header, x)}
    d["name"] = name
    return d


multi_select = pn.widgets.MultiSelect(
    name="Recordings",
    value=[],
    options=[],
    size=8,
)

str_pane = pn.pane.Str("Skills.", styles={"font-size": "12pt"})


async_flag = False
io_loop_flag = True


# raw streams
source1_raw = Stream(
    asynchronous=async_flag, ensure_io_loop=io_loop_flag
)  # asynchronous=async_flag, ensure_io_loop=io_loop_flag)
source2_raw = Stream(
    asynchronous=async_flag, ensure_io_loop=io_loop_flag
)  # asynchronous=async_flag, ensure_io_loop=io_loop_flag)
source3_raw = Stream(
    asynchronous=async_flag, ensure_io_loop=io_loop_flag
)  # asynchronous=async_flag, ensure_io_loop=io_loop_flag)
source4_raw = Stream(
    asynchronous=async_flag, ensure_io_loop=io_loop_flag
)  # asynchronous=async_flag, ensure_io_loop=io_loop_flag)

raw_source = [source1_raw, source2_raw, source3_raw, source4_raw]


# create Dataframes for mean values
source1 = source1_raw.map(lambda x: x.mean(axis=0)).map(partial(create_dict, name=1))
source2 = source2_raw.map(lambda x: x.mean(axis=0)).map(partial(create_dict, name=2))
source3 = source3_raw.map(lambda x: x.mean(axis=0)).map(partial(create_dict, name=3))
source4 = source4_raw.map(lambda x: x.mean(axis=0)).map(partial(create_dict, name=4))
mean_sources = [source1, source2, source3, source4]
combined_mean = source1.zip(source2, source3, source4)


source_df = Stream(asynchronous=async_flag, ensure_io_loop=io_loop_flag)
combined = source1_raw.zip(source2_raw, source3_raw, source4_raw)

combined_list = []
combined_list_stream = combined.sink(combined_list.append)

rr = source_df.sink(partial(add_option, multiselect=multi_select))

recordings = source_df.sink_to_list()

combined.disconnect(
    combined_list_stream,
)
record_flag = False
save_pd = None


def add_index(a: np.array):
    ra = np.concatenate(
        (np.array([range(0, a.shape[0])]).T, a),
        axis=1,
    )
    return ra


def calculate_dpgmm(frame_list, regress_columns=[0, 1, 2, 5, 6, 7]):
    print("in DPGMM")
    # extract the dataframes for all 4 measurements, but for

    dats_list = [calc_x_y_array(augment_data(frame)) for frame in frame_list]

    dat_aligned = []
    x_n, y_n = align_two_arrays(
        dats_list[0][:, regress_columns],
        dats_list[1][:, regress_columns],
        add_index=False,
    )
    dat_aligned.append(x_n)
    if len(dats_list) > 2:
        for el in dats_list[1:]:
            _, y_n = align_two_arrays(x_n, el[:, regress_columns], add_index=False)
            dat_aligned.append(y_n)

    for i, dats in enumerate(dat_aligned):
        dat_aligned[i] = add_index(dats)
    dats = np.vstack(dat_aligned)
    alder = BayesianGaussianMixture(
        n_components=10,
        covariance_type="full",
        n_init=3,
        weight_concentration_prior_type="dirichlet_distribution",
    ).fit(dats)

    # Select based upon occurance.
    # If occurance is < threas * mean occurance, drop it
    threas = 0.5
    selector, counts = np.unique(alder.predict(dats), return_counts=True)
    mask = 100 * counts / dats.shape[0] < (100 / len(selector)) * threas
    selector = selector[~mask]
    print(selector)
    print(alder.means_.shape, alder.means_[selector].shape)
    print(alder.covariances_.shape, alder.covariances_[selector].shape)
    return alder.means_[selector], alder.covariances_[selector], dats


def extract_contact_establishers(means, covariances, axes=[3]):
    ara = means[means[:, 0].argsort()]
    extracted_means = ara[:, axes].flatten()
    print("extracted means")
    print(extracted_means)
    means_thresh = 2.0
    return (
        ara[np.abs(extracted_means) > means_thresh],
        [np.abs(extracted_means) > means_thresh],
        ara,
    )


def create_suggestions(ara, selection):
    contact_establish_skill = np.diff(selection, prepend=False)
    print_list = []
    return print_list


def get_singular_plot(data, name, reset_index=False):
    xplot = "ts"
    data_copy = data.copy()
    if reset_index:
        data_copy["ts"] = data_copy["ts"] - data_copy["ts"][0]
        data_copy.reset_index()
    return [hv.Curve(data_copy, xplot, f, label=str(name)) for f in ["fx", "fy", "fz"]]


def calc_x_y_array(F: np.array):
    x1 = 0.38
    x2 = 0.38  #  + 0.05
    h = 0.01
    # y = x1/2 * ( (F1z + F2z - F3z - F4z) + h*(F1y + F2y + F3y + F4y) ) / (F1z + F2z + F3z + F4z)
    # x = x2/2 * ( (F1z - F2z - F3z + F4z) + h*(F1x + F2x + F3x + F4x) ) / (F1z + F2z + F3z + F4z)
    # F.shape: [4,7] < SensorNo, <ts,x,y,z,mx,my,mz> >
    ze = 2
    F_sum = F.sum(axis=1)
    Fxt = F_sum[:, 0]
    Fyt = F_sum[:, 1]
    Fzt = F_sum[:, 2]
    y_a = (
        1
        / Fzt
        * (h * Fyt + x1 / 2 * (F[:, 0:2, ze].sum(axis=1) - F[:, 2:, ze].sum(axis=1)))
        + x2 / 2
    )
    x_a = (
        1
        / Fzt
        * (h * Fxt + x2 / 2 * (F[:, 0::3, ze].sum(axis=1) - F[:, 1:3, ze].sum(axis=1)))
        + x1 / 2
    )

    Ft = np.linalg.norm([Fxt, Fyt, Fzt], axis=0)
    ang, mag = calc_angle_mags(np.array([Fxt, Fyt, Fzt]))

    return np.stack((x_a, y_a, Ft, ang, mag, Fxt, Fyt, Fzt), axis=-1)


def calc_x_y(F: np.array):
    x1 = 0.38
    x2 = 0.38  #  + 0.05
    h = 0.01
    # y = x1/2 * ( (F1z + F2z - F3z - F4z) + h*(F1y + F2y + F3y + F4y) ) / (F1z + F2z + F3z + F4z)
    # x = x2/2 * ( (F1z - F2z - F3z + F4z) + h*(F1x + F2x + F3x + F4x) ) / (F1z + F2z + F3z + F4z)
    # F.shape: [4,7] < SensorNo, <ts,x,y,z,mx,my,mz> >
    ze = 2
    Fxt, Fyt, Fzt = [np.sum(F[:, i]) for i in range(0, 3)]

    y = 1 / Fzt * (h * Fyt + x1 / 2 * (F[0:2, ze].sum() - F[2:, ze].sum()))
    x = 1 / Fzt * (h * Fxt + (x2 / 2) * (F[0::3, ze].sum() - F[1:3, ze].sum()))
    ang, mag = calc_angle_mags(np.array([Fxt, Fyt, Fzt]))

    return np.array(
        [
            x + x2 / 2,
            y + x1 / 2,
            np.linalg.norm([Fxt, Fyt, Fzt]),
            ang,
            mag,
            Fxt,
            Fyt,
            Fzt,
        ]
    )


def calc_angle_mags(ar: np.array):
    Fxt = ar[0]
    Fyt = ar[1]
    Fzt = ar[2]
    mag = np.sqrt(Fxt**2 + Fyt**2)
    ang = (np.pi / 2.0) - np.arctan2(Fxt / mag, Fyt / mag)
    return ang, mag


def cut_off_dfs(dfd: dict):
    """cuts off the dataframes in a dict to the shortest one"""
    # TODO values need to be sorted
    min_len = min([df.shape[0] for df in dfd.values()])
    return {key: dfd[key][-min_len:, :] for key in dfd.keys()}


def augment_data(data):
    """This seperates out one dataframe into several numpy arrays as dict according to the column 'name'
    Also synchronizes the length by cutting longer dataframes

    Args:
        data (DataFrame): dataframe with column name

    Returns:
        dict: dict with numpy arrays
    """
    names = data["name"].unique()
    DataFrameDict = {elem: pd.DataFrame() for elem in names}
    for key in DataFrameDict.keys():
        DataFrameDict[key] = data[["fx", "fy", "fz"]][data.name == key].to_numpy()
    DataFrameDict = cut_off_dfs(DataFrameDict)
    dfk = np.stack(list(DataFrameDict.values()), axis=1)
    return dfk


def get_plot(data):
    from holoviews import dim

    print("start getplot")
    names = data["name"].unique()

    dfk = augment_data(data)
    # xyf = np.array([calc_x_y(dff) for dff in dfk])
    xyf = calc_x_y_array(dfk)
    cut_off = np.quantile(xyf[:, 2], 0.5)
    xyf_c = xyf[xyf[:, 2] > cut_off, :]

    rr = []
    for name in names:
        data_selected = data.loc[data["name"] == name][["ts", "fx", "fy", "fz"]]
        data_selected.reset_index()
        rr.append(hv.Overlay(get_singular_plot(data, name)).opts(width=400, height=300))
    rr.append(
        hv.Scatter(xyf_c[:, 0:3], vdims=["y", "F"]).opts(
            color="r",
            size=dim("F") * 2,
            xlim=(-0.5, 0.5),
            ylim=(-0.5, 0.5),
            shared_axes=False,
            width=800,
            height=500,
        )
        * hv.VectorField((xyf_c[:, 0], xyf_c[:, 1], xyf_c[:, 3], xyf_c[:, 4])).opts(
            color="green",
            rescale_lengths=True,
            alpha=0.5,
            pivot="tail",
            width=800,
            height=500,
        )
    )
    hist_data = np.histogram(xyf[:, 2])
    rr.append(
        hv.Histogram(hist_data).opts(shared_axes=False, width=800, height=500)
        * hv.VLine(cut_off).opts(
            color="red", line_width=4, shared_axes=False, width=800, height=500
        )
    )
    return hv.Layout(rr).cols(2).opts(shared_axes=False, width=1600, height=800)


def rearange(tuple_list: list):
    rows = len(tuple_list)
    cols = len(tuple_list[0])
    stacked_list = []
    for col_idx in range(cols):
        to_stack_list = []
        for row_idx in range(rows):
            a = tuple_list[row_idx][col_idx]
            to_stack_list.append(a)
        stacked_list.append(np.vstack(to_stack_list))
    return np.array(stacked_list)


def align_two_arrays(x, y, start_column=0, add_index=False):
    distance, path = fastdtw(x[:, start_column:], y[:, start_column:], dist=euclidean)
    new_x_1 = np.array([p[0] for p in path])
    new_x_2 = np.array([p[1] for p in path])
    if add_index:
        a = np.concatenate(
            (
                np.array([range(0, x[:, start_column:][new_x_1].shape[0])]).T,
                x[:, start_column:][new_x_1],
            ),
            axis=1,
        )
        b = np.concatenate(
            (
                np.array([range(0, y[:, start_column:][new_x_2].shape[0])]).T,
                y[:, start_column:][new_x_2],
            ),
            axis=1,
        )
    else:
        a = x[:, start_column:][new_x_1]
        b = y[:, start_column:][new_x_2]

    return a, b


def on_click(event):
    global record_flag, combined, combined_list_stream, save_pd, combined_list, sdf, plot_panel, mean_sources, raw_source
    print("clicked")
    print("record_flag: ", record_flag)
    if not record_flag:
        print("try to connect")
        combined.connect(combined_list_stream)
        # for ms, rs in zip(mean_sources, raw_source):
        #     ms.disconnect(rs)
        record_flag = True
    else:
        combined.disconnect(combined_list_stream)
        print("convert")
        ll3_aranged = rearange(combined_list)
        print(ll3_aranged)
        # save_pd = pd.DataFrame(ll3_aranged)
        # save_pd = pd.DataFrame(itertools.chain.from_iterable(ll3_aranged))
        save_pd = pd.concat(
            pd.DataFrame(r, columns=header).assign(name=i)
            for i, r in enumerate(ll3_aranged)
        )
        save_pd.reset_index()
        # df_panel.object = save_pd
        print(save_pd)
        source_df.emit(save_pd, asynchronous=async_flag)
        print("finish on click")
        combined_list.clear()
        # for ms, rs in zip(mean_sources, raw_source):
        #     ms.connect(rs)
        record_flag = False


def create_dial(name, width=200):
    saving = defaultdict(lambda: "yellow", {"fx": "red", "fy": "green", "fz": "blue"})
    return pn.indicators.Dial(
        name=name,
        value=10,
        bounds=(-20, 20),
        format="{value:.2f} N",
        width=width,
        default_color=saving[name],
    )


def indicator_callback(df, dials, element_names):
    for i, (dial, element_name) in enumerate(zip(dials, element_names)):
        for element in zip(dial, element_name):
            element[0].value = bound_value(-20.0, 20.0, df[i][element[1]])


dials_1 = tuple(zip(*[(create_dial(b), b) for b in ["fx", "fy", "fz"]]))
dials_2 = tuple(zip(*[(create_dial(b), b) for b in ["fx", "fy", "fz"]]))
dials_3 = tuple(zip(*[(create_dial(b), b) for b in ["fx", "fy", "fz"]]))
dials_4 = tuple(zip(*[(create_dial(b), b) for b in ["fx", "fy", "fz"]]))

deb2 = combined_mean.sink(
    partial(
        indicator_callback,
        dials=[dials_1[0], dials_2[0], dials_3[0], dials_4[0]],
        element_names=[dials_1[1], dials_2[1], dials_3[1], dials_4[1]],
    )
)


# ------------
# Checks


def force_threshold(df, dim, thresh):
    ff = sum([abs(element[dim]) for element in df])
    if ff > thresh:
        return True
    else:
        return False


def bound_value(low, high, value):
    return max(low, min(high, value))


# ------------


# ------------
# Prints


def analyze_print(element):
    print(f"{type(element)}: {element}")


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


def print_event(event):
    print("click!", flush=True)


# ------------


def on_analyze(event, hv_panel):
    print(f"Recordings len: {len(recordings)}")
    names = recordings[0]["name"].unique()
    ele = [get_singular_plot(data, 0, reset_index=False) for data in recordings]
    layout = hv.Overlay([el[2] for el in ele])
    layout.opts(width=900, height=300)
    hv_panel.object = layout


def select_submatrix(cov_matrix: np.array, skill: int, axes: list):
    return cov_matrix[skill, axes[0] : axes[1], axes[0] : axes[1]]


def build_single_frame_dict(name: str, pl: list, ol: list, parent: str):
    o = {"orientation": {k: v for k, v in zip("wxyz", ol)}}
    s = {"parent": parent, "style": "none"}
    p = {"position": {k: v for k, v in zip("xyz", pl)}}
    dg = {name: {**o, **s, **p}}
    return dg


def write_out_ymls(means_sorted, selection):
    import yaml

    def_ori = [1, 0, 0, 0]
    def_z = 0
    def_parent = "table_zero"

    bs = partial(build_single_frame_dict, ol=def_ori, parent=def_parent)

    range_ar = np.arange(len(selection[0]))
    dl = {}
    for sno in range_ar[selection[0]]:
        name = f"skill_{sno}"
        pl = [float(means_sorted[sno, 1]), float(means_sorted[sno, 2]), def_z]
        dl.update(bs(name=name, pl=pl))

    dd = {"frames": dl}
    with open("frames.yaml", "w") as f:
        yaml.dump(dd, f, default_flow_style=False)


def analyzer(event, hv_panel):
    print(f"Recordings len: {len(recordings)}")
    ms, cs, aligned_data = calculate_dpgmm(recordings)
    print("----\n shapes \n")
    print(ms.shape)
    print(cs.shape)
    print("FZ Stats")
    print_stats(ms, cs, [4, 6])
    _, selection, means_sorted = extract_contact_establishers(ms, cs)
    write_out_ymls(means_sorted, selection)

    eli_list = []
    range_ar = np.arange(len(selection[0]))
    for sno in range_ar[selection[0]]:
        cvs = np.diag(select_submatrix(cs, sno, [1, 3]))
        eli = hv.Ellipse(means_sorted[sno, 1], means_sorted[sno, 2], tuple(cvs.T * 10))
        eli_text = hv.Text(means_sorted[sno, 1], means_sorted[sno, 2], f"S{sno}")
        eli_list.append(eli)
        eli_list.append(eli_text)

    l_text = [
        hv.Text(means_sorted[sno, 0], means_sorted[sno, 3], f"S{sno}")
        for sno in range_ar[selection[0]]
    ]

    #  t, x, y, Ft, Fx, Fy, Fz
    layout = hv.Overlay(
        [
            hv.Scatter(aligned_data[:, [0, 3]]),
            hv.Scatter(ms[:, [0, 3]]).opts(marker="triangle", color="red", size=10),
            *l_text,
        ]
    ).opts(width=600, height=800)

    layout += hv.Overlay(
        [
            hv.Scatter(aligned_data[:, [1, 2, 3]], vdims=["y", "F"]).opts(
                color=dim("F"), cmap="Viridis", alpha=dim("F")
            ),
            hv.Scatter(means_sorted[range_ar[selection[0]], 1:3]).opts(
                marker="triangle", color="red", size=10
            ),
            *eli_list,
        ]
    ).opts(width=1000, height=800, ylim=(-0.5, 0.5), xlim=(-0.5, 0.5))

    layout.opts(shared_axes=False, width=1600, height=800)
    hv_panel.object = layout


analyze_panel = pn.panel(hv.Curve(None), width=1600, height=400)
align_panel = pn.panel(hv.Curve(None), width=1600, height=800)


# ------------
# IO stuff


def save_list(event):
    save_to_csv(recordings)


def load_list(event):
    global recordings, multi_select

    files = glob.glob("Recording*.csv")
    print(f"Loading {files}")
    recordings = [pd.read_csv(f) for f in files]
    for r in recordings:
        r["ts"] = pd.to_datetime(r["ts"], unit="s")
    multi_select.options = files
    print(recordings)


# ------------


plot_panel = pn.pane.Streamz(source_df.map(get_plot))


# ------------
# Config
reset_topics = [f"/sensor_0{t}/ati_force_sensor/set_sw_bias" for t in range(1, 5)]
topic_list = [f"/sensor_0{t}/ati_force_sensor/ati_force_torque" for t in range(1, 5)]
# ------------


def emit_stuff(al):
    source1_raw.emit(al[0], asynchronous=async_flag)
    source2_raw.emit(al[1], asynchronous=async_flag)
    source3_raw.emit(al[2], asynchronous=async_flag)
    source4_raw.emit(al[3], asynchronous=async_flag)


# ------------
# ROS connections
rust_subscriber.start_node()
rscall = ResetCaller(reset_topics)
rscall.call_reset(None)

subscriber_thread = Thread(
    target=rust_subscriber.start_subscriber,
    kwargs={
        "buf_size": 10000,
        "dur": 1.0,
        "topic_names": topic_list,
        "pyfun": emit_stuff,
    },
    daemon=False,
)
subscriber_thread.start()


# ------------
# Button Definitions
save_button = pn.widgets.Button(name="Save data to csv", button_type="primary")
save_button.on_click(save_list)

load_button = pn.widgets.Button(name="Load data from files", button_type="primary")
load_button.on_click(load_list)

analyze_button = pn.widgets.Button(
    name="Plot and analyze selected data", button_type="primary"
)
analyze_button.on_click(partial(analyzer, hv_panel=align_panel))
analyze_button.on_click(partial(on_analyze, hv_panel=analyze_panel))

button = pn.widgets.Button(name="start stream", button_type="primary")
button.on_click(on_click)

reset_button = pn.widgets.Button(name="reset FT", button_type="primary")

reset_button.on_click(rscall.call_reset)

if debug:
    button.on_click(print_event)
    analyze_button.on_click(print_event)
    reset_button.on_click(print_event)




# ------------
# Layout
layout = pn.Row(
    pn.Column(
        pn.Row(button, reset_button),
        multi_select,
        pn.Row(analyze_button, save_button, load_button),
        pn.Row(*dials_1[0]),
        pn.Row(*dials_2[0]),
        pn.Row(*dials_3[0]),
        pn.Row(*dials_4[0]),
    ),
    pn.Column(
        plot_panel,
        analyze_panel,
        align_panel,
    ),
    str_pane,
)
# ------------


pn.serve(
    layout,
    threaded=True,
    show=True,
)
