#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <boost/filesystem.hpp>

class PCLSaverNode
{
public:
    PCLSaverNode(std::string topic_pcl, std::string save_dir);

private:
    void pclCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

    ros::NodeHandle nh;
    ros::Subscriber pcl_sub;
    int pcl_counter;
    std::string save_dir;
};