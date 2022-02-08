#include <iostream>
#include <memory>
#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <torch/script.h>
#include <torch/torch.h>

// #define DEFAULT_HEIGHT 720
// #define DEFAULT_WIDTH 1280
// #define IMG_SIZE 512



int main() {
    int deviceID = 2;
    int apiID = cv::CAP_ANY;
    cv::VideoCapture cap;
    cv::Mat frame;
    cap.open(deviceID, apiID);
    
    if(!cap.isOpened()) {
        std::cerr << "\nCannot open video\n";
    }

    std::cout << "\nPress spacebar to terminate\n";
    
    for(;;) {
        cap.read(frame);
        if(frame.empty()) {
            std::cerr << "\nError:Blank Frame\n";
        }
        cv::Point p1(0, 300), p2(640, 300);
        cv::Point p3(0, 200), p4(640, 200);
        int thickness = 10;
        cv::line(frame, p1, p2, cv::Scalar(255, 0, 0), thickness, cv::LINE_8);
        cv::line(frame, p3, p4, cv::Scalar(255, 0, 0), thickness, cv::LINE_8);
        cv::imshow("video", frame);

        char key = cv::waitKey(1);

        if(key == ' ') {
            cv::Mat croped, binaryImg, invertImg;
            cv::Mat croped_col = frame(cv::Range(200, 300), cv::Range(0, 640));
            cv::imshow("croped", croped_col);
            cv::cvtColor(croped_col, croped, cv::COLOR_BGR2GRAY);
            cv::threshold(croped, binaryImg, 80, 255, cv::THRESH_BINARY);
            cv::bitwise_not(binaryImg, invertImg);
            cv::imshow("Inverse Image", invertImg);
            for(int i = 0; i < 7; i++) {
                std::ostringstream name;
                name << "../check/" << i << ".png";
                cv::Mat number = invertImg(cv::Range(0, croped.rows), cv::Range(i*91, i*91+91));
                cv::imwrite(name.str(), number);
            }



        }
        else if (key == 'q' || key == 'Q') {
            break;
        }        
    }
}