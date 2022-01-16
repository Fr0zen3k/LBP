//
// Created by jankr on 11-Jan-22.
//

#ifndef LBP_SEGMENTATION_H
#define LBP_SEGMENTATION_H

#include "Perceptron.h"
#include <opencv2/opencv.hpp>

#include "Frame.h"

class Segmentation {
public:
    Segmentation(const Perceptron& p);
    virtual ~Segmentation();

    void ProcessFrame(Frame& frame);

private:
    Perceptron m_Perceptron;

};


#endif //LBP_SEGMENTATION_H
