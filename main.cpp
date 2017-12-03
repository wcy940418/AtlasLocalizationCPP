//
//  main.cpp
//  apriltag
//
//  Created by Yifei Wang on 11/23/17.
//  Copyright © 2017 Chaoyue Wang. All rights reserved.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "apriltag.h"
#include "tag36h11.h"
#include "common/homography.h"

using namespace std;
using namespace cv;

#define fx 3.4861838942925704e+02
#define fy 3.4861838942925704e+02
#define cx 3.1950000000000000e+02
#define cy 1.7950000000000000e+02

struct Tag {
    int id;
    Mat trans;
    Point center;
    Tag(int id, Mat trans, Point center):center(center), id(id), trans(trans) {}
};

int main(int argc, const char * argv[]) {
    VideoCapture gopro;
    double tag_size = 0.08;
    int fontface = CV_FONT_HERSHEY_SIMPLEX;
    double fontscale = 0.5;
    gopro.open(1);
    if (!gopro.isOpened()) {
        cerr << "Couldn't open camera" << endl;
        return -1;
    }
    gopro.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    gopro.set(CV_CAP_PROP_FRAME_HEIGHT, 360);
    gopro.set(CV_CAP_PROP_FPS, 60.0);
    cout << "Camera successfully opened" << endl;
    cout << "Actual resolution: "
    << gopro.get(CV_CAP_PROP_FRAME_WIDTH) << "x"
    << gopro.get(CV_CAP_PROP_FRAME_HEIGHT) << " FPS: "
    << gopro.get(CV_CAP_PROP_FPS) << endl;
    
    apriltag_family_t *tf = tag36h11_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->nthreads = 4;
    td->quad_decimate = 1.0;
    td->quad_sigma = 0.5;
    
    Mat frame, gray, display;
    
    while (true) {
        gopro >> frame;
        vector<Tag> tags;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        image_u8_t img = {
            .width = gray.cols,
            .height = gray.rows,
            .stride = gray.cols,
            .buf = gray.data
        };
        zarray_t *detections = apriltag_detector_detect(td, &img);
        
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);
            line(frame, Point(det->p[0][0], det->p[0][1]),
                 Point(det->p[1][0], det->p[1][1]),
                 Scalar(0, 0xff, 0), 2);
            line(frame, Point(det->p[0][0], det->p[0][1]),
                 Point(det->p[3][0], det->p[3][1]),
                 Scalar(0, 0, 0xff), 2);
            line(frame, Point(det->p[1][0], det->p[1][1]),
                 Point(det->p[2][0], det->p[2][1]),
                 Scalar(0xff, 0, 0), 2);
            line(frame, Point(det->p[2][0], det->p[2][1]),
                 Point(det->p[3][0], det->p[3][1]),
                 Scalar(0xff, 0, 0), 2);
            
            matd_t *pose = homography_to_pose(det->H, fx, fy, cx, cy);
            Mat trans(4, 4, CV_64F, pose->data);
            tags.push_back(Tag(det->id, trans, Point(det->c[0], det->c[1])));
        }
        Mat ref;
        bool has_ref = false;
        for (auto &t:tags) {
            if (t.id == 0) {
                ref = t.trans.inv(DECOMP_SVD);
                has_ref = true;
                break;
            }
        }
        if (has_ref) {
            for (auto &t:tags) {
                if (t.id == 0) {
                    continue;
                }
                Mat rel_trans = ref * t.trans;
                double dist = 0;
                double x, y, z;
                x = rel_trans.at<double>(0, 3) * tag_size / 2;
                y = rel_trans.at<double>(1, 3) * tag_size / 2;
                z = rel_trans.at<double>(2, 3) * tag_size / 2;
                dist = sqrt(x * x + y * y + z * z);
                vector<string> info;
                info.push_back("id: " + to_string(t.id));
                info.push_back("dist: " + to_string(dist));
                info.push_back("x: " + to_string(x));
                info.push_back("y: " + to_string(y));
                info.push_back("z: " + to_string(z));
                for (int i = 0; i < info.size(); ++i) {
                    int baseline;
                    Size textsize = getTextSize(info[i], fontface, fontscale, 1, &baseline);
                    putText(frame, info[i], Point(t.center.x,
                                               t.center.y+ i * textsize.height),
                            fontface, fontscale, Scalar(0xff, 0x99, 0), 1);
                }
               
            }
        }
        resize(frame, display, Size(0,0), 2.0f, 2.0f);
        imshow("Apriltag", display);
        char key = waitKey(1);
        if (key == 'q') {
            return 0;
        } else if (key == 's') {
            imwrite("img.jpg", frame);
        }
    }
    return 0;
}
