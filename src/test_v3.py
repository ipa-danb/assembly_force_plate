from streamz import Stream
import pandas as pd
from streamz.dataframe import DataFrame
import rospy
from geometry_msgs.msg import WrenchStamped


rospy.init_node(name="test")


def convert_force_to_df(msg=None):
    rev = {}
    rev["ts"] = pd.Timestamp(msg.header.stamp.to_sec(), unit="s")
    rev["fx"] = [msg.wrench.force.x]  # /1000000]
    rev["fy"] = [msg.wrench.force.y]  # /1000000]
    rev["fz"] = [msg.wrench.force.z]  # /1000000]
    rev["tx"] = [msg.wrench.torque.x]  # /1000000]
    rev["ty"] = [msg.wrench.torque.y]  # /1000000]
    rev["tz"] = [msg.wrench.torque.z]  # /1000000]
    reval = pd.DataFrame(rev)
    reval.set_index("ts", inplace=True)
    return reval


def convert_force_to_dict(msg=None):
    rev = {}
    rev["ts"] = pd.Timestamp(msg.header.stamp.to_sec(), unit="s")
    rev["fx"] = [msg.wrench.force.x]  # /1000000]
    rev["fy"] = [msg.wrench.force.y]  # /1000000]
    rev["fz"] = [msg.wrench.force.z]  # /1000000]
    rev["tx"] = [msg.wrench.torque.x]  # /1000000]
    rev["ty"] = [msg.wrench.torque.y]  # /1000000]
    rev["tz"] = [msg.wrench.torque.z]  # /1000000]
    return rev


def empty_df(msg=None):
    rev = {}
    if msg is not None:
        rev["ts"] = pd.Timestamp(msg.header.stamp.to_sec(), unit="s")
        rev["fx"] = [msg.wrench.force.x]  # /1000000]
        rev["fy"] = [msg.wrench.force.y]  # /1000000]
        rev["fz"] = [msg.wrench.force.z]  # /1000000]
        rev["tx"] = [msg.wrench.torque.x]  # /1000000]
        rev["ty"] = [msg.wrench.torque.y]  # /1000000]
        rev["tz"] = [msg.wrench.torque.z]  # /1000000]
    else:
        rev["ts"] = pd.Timestamp(1686923414098406572, unit="s")
        rev["fx"] = [0]
        rev["fy"] = [0]
        rev["fz"] = [0]
        rev["tx"] = [0]
        rev["ty"] = [0]
        rev["tz"] = [0]
    reval = pd.DataFrame(rev)
    reval.set_index("ts", inplace=True)
    return reval


source = Stream()

sdf = DataFrame(example=empty_df())
ll = []  # sdf.stream.sink_to_list()
ll2 = source.sink_to_list()

count = 0

t = rospy.Time.now()


def test_callback(msg):
    global count
    global t
    # data = convert_force_to_df(msg)
    # ll.append(msg)
    # sdf.emit(data)
    source.emit(convert_force_to_dict(msg))
    count += 1
    if count % 2000 == 0:
        print(2000 / (rospy.Time.now() - t).to_sec())
        # print(pd.DataFrame(ll2).set_index("ts").tail())
        sdf.emit(pd.DataFrame(ll2).set_index("ts"))
        t = rospy.Time.now()


rospy.Subscriber("wrench", WrenchStamped, test_callback)


rospy.spin()
