//
// Created by jankr on 11-Jan-22.
//

#include "Frame.h"

Frame::Frame(const cv::Mat &data): m_FrameData(data) {
    ConvertToLBP();
}

Frame::~Frame() = default;

void Frame::LoadNext(const cv::Mat &data) {
    m_FrameData = data;
    ConvertToLBP();
}

void Frame::Show(const std::string &window) {
    cv::imshow(window, m_FrameData);
}

void Frame::ConvertToLBP() {
    std::vector<unsigned char> grayscale;

    int width = m_FrameData.cols / 3;
    int height = m_FrameData.rows;

    uchar* data = m_FrameData.data;

    for(int i = 0; i < width * height * 3; i += 3) {
        grayscale.emplace_back((data[i] + data[i+1] + data[i+2]) / 3);
    }

    for(int i = 0; i < width * height; i++) {
        int x = i % width;
        int y = i / width;

        uint8_t pattern = 0;

        if(x == 0 && y == 0) {
            pattern |= 0;
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= grayscale[y*width + x + 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y+1)*width + x + 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y+1)*width + x] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= 0;
        }
        else if(x == 0) {
            pattern |= grayscale[(y-1)*width + x] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y-1)*width + x + 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[y*width + x + 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y+1)*width + x + 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y+1)*width + x] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= 0;
        }
        else if(y == 0) {
            pattern |= 0;
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= grayscale[y*width + x + 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y+1)*width + x + 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y+1)*width + x] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y+1)*width + x - 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[y*width + x - 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= 0;
        }
        else if(x == width - 1 && y == height - 1) {
            pattern |= grayscale[(y-1)*width + x] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= grayscale[y*width + x - 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y-1)*width + x - 1] > grayscale[y*width + x];
        }
        else if(x == width - 1) {
            pattern |= grayscale[(y-1)*width + x] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= grayscale[(y+1)*width + x] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y+1)*width + x - 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[y*width + x - 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y-1)*width + x - 1] > grayscale[y*width + x];
        }
        else if(y == height - 1) {
            pattern |= grayscale[(y-1)*width + x] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y-1)*width + x + 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[y*width + x + 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= 0;
            pattern <<= 1;
            pattern |= grayscale[y*width + x - 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y-1)*width + x - 1] > grayscale[y*width + x];
        }
        else {
            pattern |= grayscale[(y-1)*width + x] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y-1)*width + x + 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[y*width + x + 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y+1)*width + x + 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y+1)*width + x] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y+1)*width + x - 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[y*width + x - 1] > grayscale[y*width + x];
            pattern <<= 1;
            pattern |= grayscale[(y-1)*width + x - 1] > grayscale[y*width + x];
        }

        m_LBP.emplace_back(pattern);
    }
}

void Frame::SetAveragePosition(int x, int y) const {
    for(int i = y - 3; i <= y + 3; i++) {
        for(int j = x - 3; j <= x + 3; j++) {
            m_FrameData.data[i * m_FrameData.cols + j * 3] = 255;
            m_FrameData.data[i * m_FrameData.cols + j * 3 + 1] = 0;
            m_FrameData.data[i * m_FrameData.cols + j * 3 + 2] = 0;
        }
    }
}
