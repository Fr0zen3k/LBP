#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <array>

#include <opencv2/opencv.hpp>
#include "stb_image.h"

#include "Perceptron.h"
#include "Frame.h"
#include "Segmentation.h"

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

std::vector<uint8_t> RGBtoLBP(const unsigned char* data, int width, int height) {
    std::vector<unsigned char> grayscale;

    for(int i = 0; i < width * height * 3; i += 3) {
        grayscale.emplace_back((data[i] + data[i+1] + data[i+2]) / 3);
    }

    std::vector<uint8_t> lbp;

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

        lbp.emplace_back(pattern);
    }

    return lbp;
}

std::vector<bool> ProcessAnnotations(unsigned char* data, int width, int height) {
    std::vector<bool> annotations;

    for(int i = 0; i < width * height * 3; i += 3) {
        annotations.emplace_back(data[i] == 255 && data[i + 1] == 255 && data[i + 2] == 255);
    }

    return annotations;
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
        // train data
        std::vector<std::array<double, 257>> x;
        std::vector<bool> y;

        for(std::string input; true; ) {
            std::cout << "Please input a relative path for training or 'done' if you inputted all training paths!" << std::endl;
            std::cin >> input;

            if (input == "done") {
                break;
            }

            // use default.txt for path to labeled training data and use it to create training dataset for a perceptron
            std::string framePath;
            std::string labelPath;

            std::ifstream defaultTxt("./" + input + "/default.txt");

            if(!defaultTxt) {
                std::cout << "Unable to open 'default.txt'!" << std::endl;
                return -1;
            }

            while(defaultTxt >> framePath && defaultTxt >> labelPath) {
                int width;
                int height;
                int comp;

                std::string path = "./";
                path.append(input);

                unsigned char* frame = stbi_load(path.append(framePath).c_str(), &width, &height, &comp, STBI_rgb);
                unsigned char* annotations = stbi_load(path.append(labelPath).c_str(), &width, &height, &comp, STBI_rgb);

                // Convert to LBP
                std::vector<uint8_t> lbp = RGBtoLBP(frame, width, height);
                std::vector<bool> labels = ProcessAnnotations(annotations, width, height);

                std::array<double, 257> crowdVector = {0.0};
                std::array<double, 257> nonCrowdVector = {0.0};

                for(size_t i = 0; i < lbp.size(); i++) {
                    if(labels[i]) {
                        crowdVector[lbp[i]] += 1.0;
                    }
                    else {
                        nonCrowdVector[lbp[i]] += 1.0;
                    }
                }

                double l1 = 0.0, l2 = 0.0;

                for(int i = 0; i < 256; i++) {
                    l1 += crowdVector[i] * crowdVector[i];
                    l2 += nonCrowdVector[i] * nonCrowdVector[i];
                }

                l1 = sqrt(l1);
                l2 = sqrt(l2);

                for(int i = 0; i < 256; i++) {
                    crowdVector[i] /= l1;
                    nonCrowdVector[i] /= l2;
                }

                crowdVector[256] = 1.0;
                nonCrowdVector[256] = 1.0;

                x.emplace_back(crowdVector);
                y.emplace_back(true);
                x.emplace_back(nonCrowdVector);
                y.emplace_back(false);

                stbi_image_free(frame);
                stbi_image_free(annotations);
            }

            defaultTxt.close();
        }

        // Train the perceptron on x, y
        Perceptron p;
        p.Train(x, y);

        // Write the weights to weights.config file in the directory of the executable
        std::string execPath = argv[0];

        while(execPath[execPath.size() - 1] != '/' || execPath[execPath.size() - 1] != '\\') {
            execPath.pop_back();
        }

        std::ofstream config(execPath + "weights.config");

        if(!config) {
            std::cout << "Failed to open a config file!" << std::endl;
            return -1;
        }

        std::array<double, 257> weights = p.GetWeights();

        for(double& weight : weights) {
            config << weight;
            config << std::endl;
        }

        config.close();

        std::cout << "Training complete!" << std::endl;
    }
    else if(std::string(argv[1]) == "-r") {
        // run
        cv::Mat frameData;
        cv::namedWindow("LBP Segmentation");
        cv::VideoCapture video((std::string(argv[2])));

        if(!video.isOpened()) {
            std::cout << "No video named: '" << std::string(argv[2]) << "'!" << std::endl;
            return -1;
        }

        Frame frame;

        // Load Perceptron from weights.config
        std::string execPath = argv[0];

        while(execPath[execPath.size() - 1] != '/' || execPath[execPath.size() - 1] != '\\') {
            execPath.pop_back();
        }

        std::ifstream config(execPath + "weights.config");

        if(!config) {
            std::cout << "Failed to open a config file!" << std::endl;
            return -1;
        }

        std::array<double, 257> weights = {0.0};

        for(double& weight : weights) {
            config >> weight;
        }

        Perceptron perceptron(weights);

        // Create segmentation using perceptron
        Segmentation segmentation(perceptron);

        while(true) {
            video >> frameData;

            if(frameData.empty()) {
                break;
            }

            // Process the frame etc
            frame.LoadNext(frameData);
            segmentation.ProcessFrame(frame);

            // Show frame
            frame.Show("LBP Segmentation");

            char c = (char)cv::waitKey(25);

            if(c == 27) {
                break;
            }
        }

        video.release();

        cv::destroyAllWindows();
    }

    return 0;
}