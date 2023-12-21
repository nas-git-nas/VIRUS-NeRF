#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <boost/filesystem.hpp>

class PCLSaver
{
public:


    void subscribe(std::string topic_pcl, std::string save_dir);

private:
    void cb_pcl(const sensor_msgs::PointCloud2ConstPtr& msg);

    ros::NodeHandle nh;
    ros::Subscriber pcl_sub;
    int pcl_counter;
    std::string save_dir = "/home/spadmin/catkin_ws_ngp/data/PCL";
};