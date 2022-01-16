//
// Created by jankr on 11-Jan-22.
//

#ifndef LBP_PERCEPTRON_H
#define LBP_PERCEPTRON_H

#include <vector>
#include <array>

class Perceptron {
public:
    Perceptron();
    Perceptron(const std::array<double, 257>& weights);
    virtual ~Perceptron();

    void Train(const std::vector<std::array<double, 257>>& x, const std::vector<bool>& y);
    bool Predict(const std::array<double, 257>& featureVector);

    [[nodiscard]] inline std::array<double, 257> GetWeights() const { return m_Weights; }

private:
    std::array<double, 257> m_Weights;
    static const double LEARNING_RATE;
};


#endif //LBP_PERCEPTRON_H
