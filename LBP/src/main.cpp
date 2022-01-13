#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <string>

std::vector<std::pair<int, int>> g_CrowdMovement;

std::vector<float> NegativeVector(std::vector<float> vec) {
    for(float& f : vec) {
        f *= -1.0f;
    }

    return vec;
}

float ScalarProduct(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float sum = 0.0f;

    for(size_t i = 0; i < vec1.size(); i++) {
        sum += vec1[i] * vec2[i];
    }

    return sum;
}

std::vector<float> VectorAddition(std::vector<float> vec1, const std::vector<float>& vec2) {
    for(size_t i = 0; i < vec1.size(); i++) {
        vec1[i] += vec2[i];
    }

    return vec1;
}

std::vector<float> Normalize(std::vector<float> vec) {
    float sum = 0.0f;

    for(float f : vec) {
        sum += powf(f, 2);
    }

    sum = sqrtf(sum);

    for(float& f : vec) {
        f /= sum;
    }

    return vec;
}

int main(int argc, char *argv[]) {
    if(argc <= 1) {
        return -1;
    }

    if(std::string(argv[1]) == "-t") {
        // train

        if(std::string(argv[2]) == "-c") {
            // Continue from last weights
        }
        else {
            // Start new weights
        }
    }
    else if(std::string(argv[1]) == "-r") {
        // run
    }

    return 0;
}