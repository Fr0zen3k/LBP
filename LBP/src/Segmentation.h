//
// Created by jankr on 11-Jan-22.
//

#ifndef LBP_SEGMENTATION_H
#define LBP_SEGMENTATION_H

#include "Perceptron.h"
#include <opencv2/opencv.hpp>

class Segmentation {
public:
    Segmentation(const Perceptron& p);
    virtual ~Segmentation();

    cv::Mat ProcessFrame(cv::Mat LBP);

private:
    Perceptron m_Perceptron;
    std::vector<std::pair<int, int>> m_IncludedSegments;

};


#endif //LBP_SEGMENTATION_H
