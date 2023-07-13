import rust_subscriber
import numpy as np
import time


def print_b(b):
    print(
        f"{np.trim_zeros(b[:,0]).shape[0]/(time.perf_counter() - ts)} --{b.shape[0]}, {time.perf_counter() - ts}"
    )
    print(b[0, :])


# import threading

a = np.zeros((500, 3))

# t1 = rust_subscriber.DataField(tres=1000)

# t1.test()

# rospy.init_node("test")
rust_subscriber.start_node()

for i in range(0, 1000):
    ts = time.perf_counter()
    b_tot = rust_subscriber.start_subscriber(
        buf_size=9000,
        dur=1,
        topic_names=["/wrench_test", "/wrench_test2", "/wrench_test3"],
    )
    print(len(b_tot))
    for i, el in enumerate(b_tot):
        print(f"---   {i}   ---")
        print_b(el)

    # ts = time.perf_counter()


# def python_callback(arg):
#     print(arg)
#     return arg + 1


# result = rust_subscriber.rust_function(python_callback, 50)
