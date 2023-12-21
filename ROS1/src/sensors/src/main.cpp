#include "sensors/pcl_saver_node.h"

int main(int argc, char** argv)
{
    std::string topic_pcl = "/pointcloud_topic";
    std::string save_dir = "/home/spadmin/catkin_ws_ngp/data/PCL";

    ros::init(argc, argv, "pcl_saver_node");
    PCLSaverNode pcl_saver_node(topic_pcl, save_dir);
    ros::spin();
    return 0;
}