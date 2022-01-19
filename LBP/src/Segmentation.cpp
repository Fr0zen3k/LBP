//
// Created by jankr on 11-Jan-22.
//

#include "Segmentation.h"

Segmentation::Segmentation(const Perceptron &p): m_Perceptron(p) {}

Segmentation::~Segmentation() = default;

void Segmentation::ProcessFrame(Frame &frame) {
    std::vector<uint8_t> lbp = frame.GetLBPData();

    uchar* frameData = frame.GetFrameData().data;
    int width = frame.GetFrameData().cols / 3;
    int height = frame.GetFrameData().rows;

    int xPasses = width / 32 + 1;
    int yPasses = height / 32 + 1;

    std::vector<bool> marked;

    for(int i = 0; i < xPasses; i++) {
        for(int j = 0; j < yPasses; j++) {
            std::array<double, 257> featureVector = {0.0};

            for(int x = i * 32; x < i * 32 + 32; x++) {
                for(int y = j * 32; y < j * 32 + 32; y++) {
                    if(x >= width || y >= height) {
                        continue;
                    }

                    featureVector[lbp[y * width + x]] += 1.0;
                }
            }

            double len = 0.0;

            for(double& f : featureVector) {
                len += f * f;
            }

            len = sqrt(len);

            for(double& f : featureVector) {
                f /= len;
            }

            featureVector[256] = 1.0;

            if(!m_Perceptron.Predict(featureVector)) {
                for(int x = i * 32; x < i * 32 + 32; x++) {
                    for(int y = j * 32; y < j * 32 + 32; y++) {
                        if(x >= width || y >= height) {
                            continue;
                        }

                        frameData[y * width + x * 3] = 0;
                        frameData[y * width + x * 3 + 1] = 0;
                        frameData[y * width + x * 3 + 2] = 0;
                    }
                }

                marked.emplace_back(false);
            }
            else {
                marked.emplace_back(true);
            }
        }
    }

    int xAvg = 0, yAvg = 0;
    int count = 0;

    for(int i = 0; i < xPasses; i++) {
        for(int j=0; j < yPasses; j++) {

            if(marked[i * xPasses + j]) {
                for(int x = i * 32; x < i * 32 + 32; x++) {
                    for(int y = j * 32; y < j * 32 + 32; y++) {
                        if(x >= width || y >= height) {
                            continue;
                        }

                        xAvg += x;
                        yAvg += y;
                        count++;
                    }
                }
            }

        }
    }

    if(count > 0) {
        xAvg /= count;
        yAvg /= count;

        frame.SetAveragePosition(xAvg, yAvg);
    }
}
