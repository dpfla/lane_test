#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from lane_test.msg import msg_lane

msg = msg_lane()
class Image_class:
    def __init__(self):
        self.bridge = CvBridge()
        
        self.line = []
        self.line_md = []
        self.line_wh = []
        self.line_ye = []
        self.lane_mode = None
        self.x1_ye = None
        self.x2_ye = None
        self.y1_ye = None
        self.y2_ye = None
        self.x1_wh = None
        self.x2_wh = None
        self.y1_wh = None
        self.y2_wh = None

    def frame_img(self, image):
        self.ori_img = image
        self.img_rst = self.ori_img[120:240, 0:320]
        self.img_ye = self.ori_img[100:240, 0:160]
        self.img_wh = self.ori_img[100:240, 160:320]
        self.img_copy = self.img_rst.copy()

    def bgr_to_gray(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def edge(self, img):
        self.bgr_to_gray(img)
        blur = cv2.GaussianBlur(img, (7, 7), 0)
        self.img_canny = cv2.Canny(blur, 150, 200)


    def get_line(self):
        self.line = cv2.HoughLinesP(self.img_canny, rho=1, theta=np.pi / 180, threshold=50, minLineLength=1,
                                    maxLineGap=30)

        if self.line is not None:
            self.lane_mode = True

            for line in self.line:
                x1, y1, x2, y2 = line[0]
                cv2.line(self.img_copy, (x1, y1), (x2, y2), [255, 0, 0], 2)
        else:
            self.lane_mode = False

    def average_slope_intercept(self):
        left_lines = []
        left_weights = []
        right_lines = []
        right_weights = []

        for line in self.line:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue

                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                length = round(np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2), 3)
                if slope < 0:
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))

        self.line_ye = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
        self.line_wh = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    def make_line_points_ye(self, y1, y2):
        if self.line_ye is None:
            return None

        slope, intercept = self.line_ye

        x1_ye = np.float(y1 - intercept) / slope
        x2_ye = np.float(y2 - intercept) / slope

        if math.isinf(x1_ye):
            x1_ye = 99

        if math.isinf(x2_ye):
            x2_ye = 99

        self.x1_ye = int(x1_ye)
        self.x2_ye = int(x2_ye)
        self.y1_ye = int(y1)
        self.y2_ye = int(y2)

    def make_line_points_wh(self, y1, y2):
        if self.line_wh is None:
            return None

        slope, intercept = self.line_wh

        x1_wh = np.float(y1 - intercept) / slope
        x2_wh = np.float(y2 - intercept) / slope

        if math.isinf(x1_wh):
            x1_wh = 99

        if math.isinf(x2_wh):
            x2_wh = 99

        self.x1_wh = int(x1_wh)
        self.x2_wh = int(x2_wh)
        self.y1_wh = int(y1)
        self.y2_wh = int(y2)

    def draw_lines(self):
        y1_1 = self.img_canny.shape[0]  # bottom of the image
        y2_1 = y1_1 * 0.6  # slightly lower than the middle
        a = 0

        pre_point_wh = (self.x1_wh, self.x2_wh, self.y1_wh, self.y2_wh)
        pre_point_ye = (self.x1_ye, self.x2_ye, self.y1_ye, self.y2_ye)

        self.make_line_points_ye(y1_1, y2_1)
        self.make_line_points_wh(y1_1, y2_1)

        if self.lane_mode is True:
            if self.x1_wh is None and self.x2_wh is None and self.y1_wh is None and self.y2_wh is None:
                self.x1_wh, self.x2_wh, self.y1_wh, self.y2_wh = pre_point_wh

            elif self.x1_ye is None and self.x2_ye is None and self.y1_ye is None and self.y2_ye is None:
                self.x1_ye, self.x2_ye, self.y1_ye, self.y2_ye = pre_point_ye

            if self.line_wh is not None and self.line_ye is None:
                if self.x1_wh == self.x2_wh:
                    msg.angle = 180 * (180.0 / np.pi)
                    a = 1
                else:
                    a = 2
                    msg.angle = np.arctan((self.y1_wh - self.y2_wh) / (self.x1_wh - self.x2_wh)) * (180.0 / np.pi)
                cv2.line(self.img_copy, (self.x1_wh - 450, self.y1_wh), (self.x2_wh - 450, self.y2_wh), [0, 255, 0], 2)

            elif self.line_wh is None and self.line_ye is not None:
                if self.x1_ye == self.x2_ye:
                    msg.angle = 180 * (180.0 / np.pi)
                    a = 3
                else:
                    a = 4
                    msg.angle = np.arctan((self.y1_ye - self.y2_ye) / (self.x1_ye - self.x2_ye)) * (180.0 / np.pi)
                cv2.line(self.img_copy, (self.x1_ye - 450, self.y1_ye), (self.x2_ye - 450, self.y2_ye), [0, 255, 0], 2)

            elif not (self.line_wh is None and self.line_ye is None):
                x1 = int((self.x1_wh + self.x1_ye) // 2)
                x2 = int((self.x2_wh + self.x2_ye) // 2)
                y1 = int((self.y1_wh + self.y1_ye) // 2)
                y2 = int((self.y2_wh + self.y2_ye) // 2)

                if x2 == x1:
                    a = 5
                    msg.angle = np.arctan(y1 / x1) * (180.0 / np.pi)


                else:
                    a = 6
                    msg.angle = np.arctan((y1 - y2) / (x1 - x2)) * (180.0 / np.pi)
 
                self.line_md.append((x1, y1, x2, y2))

                cv2.line(self.img_copy, (x1, y1), (x2, y2), [0, 255, 0], 2)

            elif self.line_wh is None and self.line_ye is None:
                pass

            self.__init__()
            rospy.loginfo("%f" %(msg.angle))
            rospy.loginfo("%d" %(a))
            


def main():
    pub = rospy.Publisher('/image_raw', Image, queue_size=1)
    pub_lane = rospy.Publisher('/data_lane', msg_lane, queue_size=1)
    rospy.init_node("lane_1")
    rate = rospy.Rate(10)
    img_now = Image_class()
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    if not cap.isOpened():
        print("open fail video")
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            img_now.frame_img(frame)
            img_now.edge(img_now.img_rst)
            # img_now.select_region(img_now.img_canny)
            img_now.get_line()
            if img_now.lane_mode == True:
                img_now.average_slope_intercept()
                img_now.draw_lines()

            frame_ = img_now.bridge.cv2_to_imgmsg(img_now.img_copy, "bgr8")
            pub.publish(frame_)
            pub_lane.publish(msg)

            k = cv2.waitKey(10)
            if k == 27:
                cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

