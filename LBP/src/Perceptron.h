//
// Created by jankr on 11-Jan-22.
//

#ifndef LBP_PERCEPTRON_H
#define LBP_PERCEPTRON_H

#include <vector>

class Perceptron {
public:
    Perceptron();
    Perceptron(const std::vector<float>& weights);
    virtual ~Perceptron();

    void AdjustWeights(const std::vector<float>& featureVector);
    bool Predict(const std::vector<float>& featureVector);

private:
    std::vector<float> m_Weights;

};


#endif //LBP_PERCEPTRON_H
