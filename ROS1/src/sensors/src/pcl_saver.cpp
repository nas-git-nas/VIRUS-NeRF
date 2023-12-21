#include "sensors/pcl_saver.h"

void PCLSaver::subscribe(std::string topic_pcl, std::string save_dir)
{
    pcl_sub = nh.subscribe(topic_pcl, 1, &PCLSaver::cb_pcl, this);


}

void PCLSaver::cb_pcl(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);

    std::string file_path = save_dir + "/pcl_" + std::to_string(pcl_counter) + ".pcd";
    pcl::io::savePCDFileASCII(file_path, cloud);
    pcl_counter++;
}