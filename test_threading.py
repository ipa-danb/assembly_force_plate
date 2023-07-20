import rust_subscriber
import numpy as np
import time
from functools import partial
from threading import Thread
from streamz import Stream
import rospy
from std_msgs.msg import Empty
from rich import print

tk = time.perf_counter()


def print_b(b):
    global tk
    print(f"thread returned in {time.perf_counter() - tk}")
    tk = time.perf_counter()


rust_subscriber.start_node()

ts = time.perf_counter()

topic_list = [f"/sensor_0{t}/ati_force_sensor/ati_force_torque" for t in range(1, 5)]

t1 = Thread(
    target=rust_subscriber.start_subscriber,
    kwargs={
        "buf_size": 1000,
        "dur": 0.5,
        "topic_names": topic_list,  # , "/wrench_test2", "/wrench_test3"],
        "pyfun": print_b,
    },
    daemon=False,
)
t1.start()

slept = 1.0
ts = time.perf_counter()


# import rospy

# rospy.spin()
while True:
    print("bibbber")
    # time.sleep(slept)
    print("Blubber")
    time.sleep(slept)
    print(f"busy, slept for {time.perf_counter() - ts}")
    ts = time.perf_counter()


# def python_callback(arg):
#     print(arg)
#     return arg + 1


# result = rust_subscriber.rust_function(python_callback, 50)
