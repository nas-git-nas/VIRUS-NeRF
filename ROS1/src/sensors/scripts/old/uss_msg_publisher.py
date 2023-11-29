#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
from std_msgs.msg import Header
from sensors.msg import USS

def talker():
    pub = rospy.Publisher('chatter', USS, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    
    while not rospy.is_shutdown():
        # hello_str = "hello world %s" % rospy.get_time()

        
        header = Header(stamp=rospy.Time.now(), frame_id="uss")
        
        uss_msg = USS(header=header, meas=22)
        

        rospy.loginfo(uss_msg.meas)
        rospy.loginfo(uss_msg.header)
        pub.publish(uss_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass