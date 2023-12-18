#include "sensors/pcl_saver_node.h"
#include <fstream>
#include <string>

PCLSaverNode::PCLSaverNode(std::string topic_pcl, std::string save_dir)
    : pcl_counter(0), save_dir(save_dir)
{
    pcl_sub = nh.subscribe(topic_pcl, 1, &PCLSaverNode::pclCallback, this);

    // if (boost::filesystem::exists(save_dir))
    // {
    //     boost::filesystem::remove_all(save_dir);
    // }
    // boost::filesystem::create_directory(save_dir);
}

void PCLSaverNode::pclCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::fromROSMsg(*msg, cloud);

    std::string file_path = save_dir + "/full" + std::to_string(pcl_counter) + ".pcd";
    pcl::io::savePCDFileASCII(file_path, cloud);
    pcl_counter++;

    ROS_INFO("Saved point cloud %d", pcl_counter);
}

void PCLSaverNode::poseCallback(const sensor_msgs::OdometryConstPtr& msg) 
{
    std::string file_path = save_dir + "/alidarPose.csv";

    p = msg.pose.pose
    pose = PCLTransformer(
        t=[p.position.x, p.position.y, p.position.z],
        q=[p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w],
    )

    // Open the file in append mode
    std::ofstream file;
    file.open(file_path, std::ios_base::app);

    // Check if the file is open
    if (!file) {
        ROS_ERR("Unable to open file %s", filename)
        return;
    }

    // Write the line to the file
    file << line << "\n";

    // Close the file
    file.close();
}

int main(int argc, char** argv)
{
    // std::string topic_pcl = topic_pcl;
    // if (ros::param::get("topic_pcl", topic_pcl)) {
    //     ROS_ERROR("Failed to get param 'topic_pcl'");
    // }

    // std::string save_dir = save_dir;
    // if (ros::param::get("save_dir", save_dir)) {
    //     ROS_ERROR("Failed to get param 'save_dir'");
    // } else {
    //     ROS_INFO("Saving point clouds to LLLLLLLLLLLLL");
    // }

    std::string topic_pcl = "/rslidar_points";
    std::string save_dir = "/home/spadmin/catkin_ws_ngp/data/PCL";

    ros::init(argc, argv, "pcl_saver_node");
    PCLSaverNode pcl_saver_node(topic_pcl, save_dir);
    ros::spin();
    return 0;
}