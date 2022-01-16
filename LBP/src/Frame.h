//
// Created by jankr on 11-Jan-22.
//

#ifndef LBP_FRAME_H
#define LBP_FRAME_H

#include <opencv2/opencv.hpp>

#include <vector>
#include <array>

class Frame {
public:
    Frame() = default;
    Frame(const cv::Mat& data);
    virtual ~Frame();

    void LoadNext(const cv::Mat& data);
    void SetAveragePosition(int x, int y) const;
    void Show(const std::string& window);
    void ConvertToLBP();

    [[nodiscard]] inline cv::Mat GetFrameData() const { return m_FrameData; }
    [[nodiscard]] inline std::vector<uint8_t> GetLBPData() const { return m_LBP; }

private:
    cv::Mat m_FrameData;
    std::vector<uint8_t> m_LBP;
};


#endif //LBP_FRAME_H
