#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from lane_test.msg import msg_lane
def callback(data):
    print("ddd")
    rospy.loginfo('scan.ranges[359]: %s',data.angle)
    if data.angle <= 90 and data.angle >= 80:   
        gos()
       
    else :
        stop()

    velocity_publisher.publish(vel_msg)
    
    
def gos():
        vel_msg.linear.x = 1
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0.5

def stop():
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0

def move1():
    while not rospy.is_shutdown():
        rospy.spin()






if __name__ == '__main__':
    rospy.init_node('move_lane', anonymous=True)
    velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber('/data_lane', msg_lane, callback)
    vel_msg = Twist()
      
  
    try:
        #Testing our function
        while(1):
            move1()

          
    except rospy.ROSInterruptException: pass
