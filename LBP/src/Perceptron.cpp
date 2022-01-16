//
// Created by jankr on 11-Jan-22.
//

#include "Perceptron.h"

#include <random>

const double Perceptron::LEARNING_RATE = 1.0;

Perceptron::Perceptron() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distribution(0.0, 1.0);

    for(double & m_Weight : m_Weights) {
        m_Weight = distribution(gen);
    }
}

Perceptron::Perceptron(const std::array<double, 257> &weights): m_Weights(weights) {

}

Perceptron::~Perceptron() {

}

void Perceptron::Train(const std::vector<std::array<double, 257>> &x, const std::vector<bool> &y) {
    size_t correct = 0;

    while(correct != y.size()) {
        for(size_t i = 0; i < x.size(); i++) {
            std::array<double, 257> currentX = x[i];

            double res = 0.0;

            if(!y[i]) {
                for(double& val : currentX) {
                    val *= -1.0;
                }
            }

            for(size_t j = 0; j < currentX.size(); j++) {
                res += m_Weights[j] * currentX[j];
            }

            if(res <= 0) {
                if(LEARNING_RATE != 1.0) {
                    for(double& val : currentX) {
                        val *= LEARNING_RATE;
                    }
                }

                for(size_t j = 0; j < m_Weights.size(); j++) {
                    m_Weights[j] += currentX[j];
                }
            }
        }
    }
}

bool Perceptron::Predict(const std::array<double, 257> &featureVector) {
    double res = 0.0;

    for(size_t i = 0; i < m_Weights.size(); i++) {
        res += m_Weights[i] * featureVector[i];
    }

    return res >= 0.0;
}
