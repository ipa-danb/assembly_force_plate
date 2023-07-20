import rospy
from std_msgs.msg import Empty
import time


class ResetCaller:
    def __init__(self, topic_names):
        self.s = [
            rospy.Publisher(
                topic_name,
                Empty,
                # queue_size=2,
            )
            for topic_name in topic_names
        ]
        print(self.s)

    def call_reset(self, event=None):
        for element in self.s:
            for i in range(0, 10):
                element.publish(Empty())


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

time.sleep(1.0)
