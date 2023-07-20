#!/usr/bin/env python
import rospy
from geometry_msgs.msg import WrenchStamped
import pandas as pd
import numpy as np
import argparse
import itertools
from turtlesim.srv import TeleportAbsolute

from functools import partial


def rep_list_until(ll: list, l: int):
    """repeats list ll until it reaches the length l

    Args:
        ll (list): list to repeat
        l (int): length

    Returns:
        list: list of length l with repeated elements from ll
    """
    return list(itertools.islice(itertools.cycle(ll), l))


class FT_Mockup:
    header = [f"{f}{dim}" for f in "FM" for dim in "xyz"]

    def __init__(
        self,
        replay_file: str = None,
        topics: list = ["wrench_test1", "wrench_test2"],
        freq: int = 1000,
        f_r: list = [10.0, 5.0, 1.0],
        f_m: list = [1.0, 1.0, 0.5],
    ):
        self.run_flag = True
        self.freq = freq
        self.iterators = []
        self.no_repeats = len(topics)
        self.df = []

        f_m_l = rep_list_until(f_m, len(topics) * 3)
        f_r_l = rep_list_until(f_r, len(topics) * 3)

        if replay_file is None:
            for i, element in enumerate(topics):
                nn = np.zeros((2, len(self.header)))
                nn[0, 0:3] = np.array(f_m_l[i * 3 : (i + 1) * 3])
                nn[1, 0:3] = np.array(f_r_l[i * 3 : (i + 1) * 3])
                self.df.append(pd.DataFrame(nn, columns=self.header, index=["m", "r"]))
                self.create_random_replay_file(self.df[-1])
        else:
            self.load_replay_file(replay_file)

        # TODO make this a rosparam
        rospy.init_node("FT_Sensor", anonymous=True)
        # TODO: Make this a rosparam
        self.publishers = [
            rospy.Publisher(n, WrenchStamped, queue_size=10) for n in topics
        ]

        self.sv_prox = [
            rospy.Service(
                f"/change_element/F{a}",
                TeleportAbsolute,
                partial(self.service, element=a),
            )
            for a in range(0, self.no_repeats)
        ]

        while True:
            self.publish_data()

    def service(self, data, element=0):
        x = data.x
        y = data.y
        z = data.theta
        fml = [x, y, z]
        self.df[element].loc[["m"], ["Fx", "Fy", "Fz"]] = fml

        self.create_random_replay_file(self.df[element], no=element)
        return []

    def load_replay_file(self, replay_file: str):
        """load a file to be replayed from csv

        Args:
            replay_file (str): file name

        Returns:
            bool: success
        """
        # TODO: actually test this for real data
        try:
            self.replay_set = pd.read_csv(replay_file)
            self.iterator = itertools.cycle(self.replay_set.iterrows())
            return True
        except FileNotFoundError:
            return False

    def create_random_replay_file(self, df, no=None):
        sec = 100

        # self.iterators = []

        ma = np.array([df.loc["m"]])
        ra = np.array([df.loc["r"]])

        total_len = int(sec * self.freq)

        data_np = ma.repeat(total_len, axis=0) + ra.repeat(
            total_len, axis=0
        ) * np.random.uniform(-1, 1, size=(total_len, 1))

        data = pd.DataFrame(
            data_np,
            columns=self.header,
        )
        self.replay_set = data
        if no is None:
            self.iterators.append(itertools.cycle(self.replay_set.iterrows()))
        else:
            self.iterators[no] = itertools.cycle(self.replay_set.iterrows())

    def create_new_wrench_msg(self, it):
        w_msg = WrenchStamped()
        data = next(it)[1]

        w_msg.header.stamp = rospy.Time.now()

        w_msg.wrench.force.x = data["Fx"]
        w_msg.wrench.force.y = data["Fy"]
        w_msg.wrench.force.z = data["Fz"]
        w_msg.wrench.torque.x = data["Mx"]
        w_msg.wrench.torque.y = data["My"]
        w_msg.wrench.torque.z = data["Mz"]

        return w_msg

    def publish_data(self):
        if not self.run_flag:
            return
        for pubs, it in zip(self.publishers, self.iterators):
            pubs.publish(self.create_new_wrench_msg(it))
        rospy.sleep(1 / self.freq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock up data replay for FT Sensor")
    parser.add_argument("--replay_file", type=str, default=None, help="file to replay")
    parser.add_argument("--topics", nargs="*", default=["wrench_test1"])
    parser.add_argument("--freq", type=int, default=1000)
    parser.add_argument("--forces_mean", nargs="*", default=[10.0, 5.0, 1.0])
    parser.add_argument("--forces_range", nargs="*", default=[1.0, 1.0, 0.5])

    args = parser.parse_args()
    print(args)
    replay_file = args.replay_file

    FT_Mockup(
        replay_file=args.replay_file,
        topics=args.topics,
        freq=args.freq,
        f_r=args.forces_range,
        f_m=args.forces_mean,
    )

    rospy.spin()
