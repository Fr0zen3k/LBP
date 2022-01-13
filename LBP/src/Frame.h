//
// Created by jankr on 11-Jan-22.
//

#ifndef LBP_FRAME_H
#define LBP_FRAME_H

#include <opencv2/opencv.hpp>

class Frame {
public:
    Frame(cv::Mat data);
    virtual ~Frame();

    void LoadNext(cv::Mat data);
    void Show();
    void RGBtoGrayscale();
    void GrayscaleToLBP();

    cv::Mat GetFrameData();

private:
    cv::Mat m_FrameData;
    cv::Mat m_Grayscale;
    cv::Mat m_LBP;
};


#endif //LBP_FRAME_H
