import os

from pose_sync import PoseSync
from time_sync import TimeSync
from rosbag_wrapper import RosbagWrapper

def readData(
    data_dir:str,
    bag_name:str,
):
    """
    Read data in one step for faster processing.
    Args:
        data_dir:
    """
    bag_wrapper = RosbagWrapper(
        bag_path=os.path.join(data_dir, bag_name),
    )
    meass_dict, times_dict = bag_wrapper.readBag(
        topics=[
            "/CAM1/color/image_raw",
            "/CAM3/color/image_raw",
            "/USS1",
            "/USS3",
            "/TOF1",
            "/TOF3",
        ],
    )
    
    



def main():
    
    data_dir = "/home/spadmin/catkin_ws_ngp/data/test"
    bag_name = "test.bag"
    poses_name = "alidarPose.csv"
    
    # topics to copy and paste fom old bag
    keep_topics = [
        "/CAM1/depth/color/points",
        "/CAM3/depth/color/points",
        "/rslidar_points",
    ]
    
    # synchronized pose topics
    ps = PoseSync(
        data_dir=data_dir,
        bag_name=bag_name,
        poses_name=poses_name,
    )
    write_topics, write_msgs = ps()
    
    # synchronized time topics
    ts = TimeSync(
        data_dir=data_dir,
        bag_name=bag_name,
    )
    replace_topics_r, replace_topics_w, replace_times_r, replace_times_w = ts()
    
    ps.newBag(
        new_bag_path=os.path.join(data_dir, bag_name.replace('.bag', '_sync.bag')),
        keep_topics=keep_topics,
        write_topics=write_topics,
        write_msgs=write_msgs,
        replace_topics_r=replace_topics_r,
        replace_topics_w=replace_topics_w,
        replace_times_r=replace_times_r,
        replace_times_w=replace_times_w,
    )


if __name__ == "__main__":
    main()