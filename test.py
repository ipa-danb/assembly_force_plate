import rust_subscriber
import numpy as np
import time
from functools import partial
from threading import Thread
from streamz import Stream
import rospy
from std_msgs.msg import Empty
from rich import print


def print_b(b):
    print(
        f"{np.trim_zeros(b[:,0]).shape[0]/(time.perf_counter() - ts)} --{b.shape[0]}, {time.perf_counter() - ts}"
    )
    print(b[0, :])


x1 = 0.38
x2 = 0.38  #  + 0.05
h = 0.01


def calc_total_force(a):
    return np.sum(a, axis=0)


def extract_element(a, el):
    rev = np.array(a)
    return rev[:, el]


def calc_x_y(F: np.array):
    # y = x1/2 * ( (F1z + F2z - F3z - F4z) + h*(F1y + F2y + F3y + F4y) ) / (F1z + F2z + F3z + F4z)
    # x = x2/2 * ( (F1z - F2z - F3z + F4z) + h*(F1x + F2x + F3x + F4x) ) / (F1z + F2z + F3z + F4z)
    # F.shape: [4,7] < SensorNo, <ts,x,y,z,mx,my,mz> >
    ze = 2
    Fxt, Fyt, Fzt = [np.sum(F[:, i]) for i in range(0, 3)]
    # y = 1 / Fzt * (h * Fyt + x1 / 2 * (F[0:2, ze].sum() - F[2:, ze].sum()))
    # x = 1 / Fzt * (h * Fxt + x2 / 2 * (F[0::3, ze].sum() - F[1:3, ze].sum()))
    # print(F[0::3, ze], F[1:3, ze])
    y = 1 / Fzt * (h * Fyt + x1 / 2 * (F[0:2, ze].sum() - F[2:, ze].sum()))
    x = 1 / Fzt * (h * Fxt + (x2 / 2) * (F[0::3, ze].sum() - F[1:3, ze].sum()))

    return np.array([x - x2 / 2, y + x1 / 2])


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

    def call_reset(self, event=None):
        for i in range(0, 10):
            for element in self.s:
                element.publish()


reset_topics = [
    "/sensor_01/ati_force_sensor/set_sw_bias",
    "/sensor_02/ati_force_sensor/set_sw_bias",
    "/sensor_03/ati_force_sensor/set_sw_bias",
    "/sensor_04/ati_force_sensor/set_sw_bias",
]

rospy.init_node("abra")

rc = ResetCaller(reset_topics)

time.sleep(1.0)

rc.call_reset()


source1_raw = Stream()
source2_raw = Stream()
source3_raw = Stream()
source4_raw = Stream()


s1 = source1_raw.map(lambda x: x.mean(axis=0))
# s1.map(lambda x: f"1:{x[1:4]}").sink(print)
s2 = source2_raw.map(lambda x: x.mean(axis=0))
# s2.map(lambda x: f"2:{x[1:4]}").sink(print)
s3 = source3_raw.map(lambda x: x.mean(axis=0))
# s3.map(lambda x: f"3:{x[1:4]}").sink(print)
s4 = source4_raw.map(lambda x: x.mean(axis=0))


flist_stream = s1.zip(s2, s3, s4)
fzlist_stream = flist_stream.map(partial(extract_element, el=[1, 2, 3]))
ft_stream = fzlist_stream.map(calc_total_force)
# ft_stream.sink(lambda x: print(" | ".join([f"{r:.3f}" for r in x])))

# now calculate x,y
xy_stream = fzlist_stream.map(calc_x_y)
xy_stream.sink(lambda x: print(" | ".join([f"{r*100:.2f}" for r in x])))


def emit_stuff(al):
    source1_raw.emit(al[0])
    source2_raw.emit(al[1])
    source3_raw.emit(al[2])
    source4_raw.emit(al[3])


rust_subscriber.start_node()

ts = time.perf_counter()

topic_list = [f"/sensor_0{t}/ati_force_sensor/ati_force_torque" for t in range(1, 5)]

t1 = Thread(
    target=rust_subscriber.start_subscriber,
    kwargs={
        "buf_size": 1000,
        "dur": 0.2,
        "topic_names": topic_list,  # , "/wrench_test2", "/wrench_test3"],
        "pyfun": emit_stuff,
    },
    daemon=False,
)

t1.start()

slept = 0.1
ts = time.perf_counter()
