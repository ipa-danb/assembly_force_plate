#!/usr/bin/env python
import rospy
from geometry_msgs.msg import WrenchStamped
import pandas as pd
import numpy as np
import argparse
import itertools


class FT_Mockup:
    header = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    def __init__(self, replay_file: str = None):
        self.run_flag = False
        if replay_file is None:
            self.create_random_replay_file()
        else:
            self.load_replay_file(replay_file)

        # TODO make this a rosparam
        rospy.init_node("FT_Sensor", anonymous=True)
        # TODO: Make this a rosparam
        self.publisher = rospy.Publisher("wrench", WrenchStamped, queue_size=10)

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

    def create_random_replay_file(self):
        rrange_F = 10.0
        rrange_M = 1.0
        sec = 100
        freq = 1000

        total_len = int(sec * freq)
        header_len = int(len(self.header) / 2)
        multiply_mask = np.hstack(
            (
                np.ones((total_len, header_len)) * rrange_F,
                np.ones((total_len, header_len)) * rrange_M,
            )
        )

        data = pd.DataFrame(
            np.random.uniform(-1, 1, size=(total_len, header_len * 2)) * multiply_mask,
            columns=self.header,
        )
        self.replay_set = data
        self.iterator = itertools.cycle(self.replay_set.iterrows())

    def create_new_wrench_msg(self):
        w_msg = WrenchStamped()
        data = self.iterator()

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
        self.publisher.publish(self.creat_new_wrench_msg)
        rospy.sleep(1 / 1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock up data replay for FT Sensor")
    parser.add_argument("--replay_file", type=str, default=None, help="file to replay")
    args = parser.parse_args()
    print(args)
    replay_file = args.replay_file

    FT_Mockup(replay_file=args.replay_file)

    rospy.spin()
