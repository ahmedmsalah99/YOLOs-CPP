#pragma once

// =====================================
// Single Image Classifier Header File
// =====================================
//
// This header defines the YOLO8Classifier class for performing image classification
// using an ONNX model. It includes necessary libraries, utility structures,
// and helper functions to facilitate model inference and result interpretation.
//
// =====================================

/**
 * @file YOLO8CLASS.hpp
 * @brief Header file for the YOLO8Classifier class, responsible for image classification
 * using an ONNX model with optimized performance for minimal latency.
 */

// Include necessary ONNX Runtime and OpenCV headers
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <unordered_map>
#include <thread>
#include <iomanip> // For std::fixed and std::setprecision
#include <sstream> // For std::ostringstream

// #define DEBUG_MODE // Enable debug mode for detailed logging

// Include debug and custom ScopedTimer tools for performance measurement
// Assuming these are in a common 'tools' directory relative to this header
#include "tools/Debug.hpp"
#include "tools/ScopedTimer.hpp"

/**
 * @brief Struct to represent a classification result.
 */
struct ClassificationResult {
    int classId{-1};        // Predicted class ID, initialized to -1 for easier error checking
    float confidence{0.0f}; // Confidence score for the prediction
    std::string className{}; // Name of the predicted class

    ClassificationResult() = default;
    ClassificationResult(int id, float conf, std::string name)
        : classId(id), confidence(conf), className(std::move(name)) {}
};


/**
 * @namespace utils
 * @brief Namespace containing utility functions for the YOLO11Classifier.
 */
namespace utils {

    // ... (clamp, getClassNames, vectorProduct, preprocessImageToTensor, drawClassificationResult utilities remain the same as previous correct version) ...
    /**
     * @brief A robust implementation of a clamp function.
     * Restricts a value to lie within a specified range [low, high].
     */
    template <typename T>
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type
    inline clamp(const T &value, const T &low, const T &high)
    {
        T validLow = low < high ? low : high;
        T validHigh = low < high ? high : low;

        if (value < validLow) return validLow;
        if (value > validHigh) return validHigh;
        return value;
    }

    /**
     * @brief Loads class names from a given file path.
     */
    std::vector<std::string> getClassNames(const std::string &path) {
        std::vector<std::string> classNames;
        std::ifstream infile(path);

        if (infile) {
            std::string line;
            while (getline(infile, line)) {
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                classNames.emplace_back(line);
            }
        } else {
            std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
        }

        DEBUG_PRINT("Loaded " << classNames.size() << " class names from " + path);
        return classNames;
    }

    /**
     * @brief Computes the product of elements in a vector.
     */
    size_t vectorProduct(const std::vector<int64_t> &vector) {
        if (vector.empty()) return 0;
        return std::accumulate(vector.begin(), vector.end(), 1LL, std::multiplies<int64_t>());
    }

    /**
     * @brief Prepares an image for model input by resizing and padding (letterboxing style) or simple resize.
     */
    inline void preprocessImageToTensor(const cv::Mat& image, cv::Mat& outImage,
                                      const cv::Size& targetShape,
                                      const cv::Scalar& color = cv::Scalar(0, 0, 0),
                                      bool scaleUp = true,
                                      const std::string& strategy = "resize") {
        if (image.empty()) {
            std::cerr << "ERROR: Input image to preprocessImageToTensor is empty." << std::endl;
            return;
        }

        if (strategy == "letterbox") {
            float r = std::min(static_cast<float>(targetShape.height) / image.rows,
                               static_cast<float>(targetShape.width) / image.cols);
            if (!scaleUp) {
                r = std::min(r, 1.0f);
            }
            int newUnpadW = static_cast<int>(std::round(image.cols * r));
            int newUnpadH = static_cast<int>(std::round(image.rows * r));

            cv::Mat resizedTemp;
            cv::resize(image, resizedTemp, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);

            int dw = targetShape.width - newUnpadW;
            int dh = targetShape.height - newUnpadH;

            int top = dh / 2;
            int bottom = dh - top;
            int left = dw / 2;
            int right = dw - left;

            cv::copyMakeBorder(resizedTemp, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
        } else { // Default to "resize"
            if (image.size() == targetShape) {
                outImage = image.clone();
            } else {
                cv::resize(image, outImage, targetShape, 0, 0, cv::INTER_LINEAR);
            }
        }
    }

    /**
     * @brief Draws the classification result on the image.
     */
    inline void drawClassificationResult(cv::Mat &image, const ClassificationResult &result,
                                         const cv::Point& position = cv::Point(10, 10),
                                         const cv::Scalar& textColor = cv::Scalar(0, 255, 0),
                                         double fontScaleMultiplier = 0.0008,
                                         const cv::Scalar& bgColor = cv::Scalar(0,0,0) ) {
        if (image.empty()) {
            std::cerr << "ERROR: Empty image provided to drawClassificationResult." << std::endl;
            return;
        }
        if (result.classId == -1) {
            DEBUG_PRINT("Skipping drawing due to invalid classification result.");
            return;
        }

        std::ostringstream ss;
        ss << result.className << ": " << std::fixed << std::setprecision(2) << result.confidence * 100 << "%";
        std::string text = ss.str();

        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = std::min(image.rows, image.cols) * fontScaleMultiplier;
        if (fontScale < 0.4) fontScale = 0.4;
        const int thickness = std::max(1, static_cast<int>(fontScale * 1.8));
        int baseline = 0;

        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
        baseline += thickness;

        cv::Point textPosition = position;
        if (textPosition.x < 0) textPosition.x = 0;
        if (textPosition.y < textSize.height) textPosition.y = textSize.height + 2; 

        cv::Point backgroundTopLeft(textPosition.x, textPosition.y - textSize.height - baseline / 3);
        cv::Point backgroundBottomRight(textPosition.x + textSize.width, textPosition.y + baseline / 2);
        
        backgroundTopLeft.x = utils::clamp(backgroundTopLeft.x, 0, image.cols -1);
        backgroundTopLeft.y = utils::clamp(backgroundTopLeft.y, 0, image.rows -1);
        backgroundBottomRight.x = utils::clamp(backgroundBottomRight.x, 0, image.cols -1);
        backgroundBottomRight.y = utils::clamp(backgroundBottomRight.y, 0, image.rows -1);

        cv::rectangle(image, backgroundTopLeft, backgroundBottomRight, bgColor, cv::FILLED);
        cv::putText(image, text, cv::Point(textPosition.x, textPosition.y), fontFace, fontScale, textColor, thickness, cv::LINE_AA);

        DEBUG_PRINT("Classification result drawn on image: " << text);
    }

}; // end namespace utils

/**
 * @class YOLO8Classifier
 * @brief Class for performing image classification using an ONNX model.
 */
class YOLO8Classifier {
public:
    /**
     * @brief Constructor to initialize the classifier with model and label paths.
     */
    YOLO8Classifier(const std::string &modelPath, const std::string &labelsPath,
                    bool useGPU = false, const cv::Size& targetInputShape = cv::Size(224, 224));

    /**
     * @brief Runs classification on the provided image.
     */
    ClassificationResult classify(const cv::Mat &image);

    /**
     * @brief Draws the classification result on the image.
     */
    void drawResult(cv::Mat &image, const ClassificationResult &result,
                    const cv::Point& position = cv::Point(10, 10)) const {
        utils::drawClassificationResult(image, result, position);
    }

    cv::Size getInputShape() const { return inputImageShape_; } // CORRECTED
    bool isModelInputShapeDynamic() const { return isDynamicInputShape_; } // CORRECTED
    int getNumClasses() const { return numClasses_; } // Get number of classes
    const std::vector<std::string>& getClassNames() const { return classNames_; } // Get class names

private:
    Ort::Env env_; // ONNX Runtime environment
    Ort::SessionOptions sessionOptions_; // Session options for ONNX Runtime
    Ort::Session session_; // ONNX Runtime session

    std::vector<std::string> inputNames_; // Input node names
    std::vector<std::string> outputNames_; // Output node names
    std::vector<std::unique_ptr<char[]>> inputNodeNameAllocatedStrings_;
    std::vector<std::unique_ptr<char[]>> outputNodeNameAllocatedStrings_;

    cv::Size inputImageShape_; // Target input shape for the model
    bool isDynamicInputShape_{false}; // Flag to indicate if the input shape is dynamic

    size_t numInputNodes_{}, numOutputNodes_{};
    int numClasses_{0};

    std::vector<std::string> classNames_{};

    void preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);
    ClassificationResult postprocess(const std::vector<Ort::Value> &outputTensors);
};

// Implementation of YOLO8Classifier constructor
YOLO8Classifier::YOLO8Classifier(const std::string &modelPath, const std::string &labelsPath,
                                   bool useGPU, const cv::Size& targetInputShape)
    : inputImageShape_(targetInputShape) {
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_CLASSIFICATION_ENV");
    sessionOptions_ = Ort::SessionOptions();

    sessionOptions_.SetIntraOpNumThreads(std::min(4, static_cast<int>(std::thread::hardware_concurrency())));
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption{};

    if (useGPU && cudaAvailable != availableProviders.end()) {
        DEBUG_PRINT("Attempting to use GPU for inference.");
        sessionOptions_.AppendExecutionProvider_CUDA(cudaOption);
    } else {
        if (useGPU) {
            std::cout << "Warning: GPU requested but CUDAExecutionProvider is not available. Falling back to CPU." << std::endl;
        }
        DEBUG_PRINT("Using CPU for inference.");
    }
    session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
    DEBUG_PRINT("ONNX model loaded from: " << modelPath);
    Ort::AllocatorWithDefaultOptions allocator;
    inputNodeNameAllocatedStrings_.clear();
    outputNodeNameAllocatedStrings_.clear();
    inputNames_.clear();
    outputNames_.clear();
    numInputNodes_ = session_.GetInputCount();
    numOutputNodes_ = session_.GetOutputCount();
    DEBUG_PRINT("Model has " << numInputNodes_ << " input nodes and " << numOutputNodes_ << " output nodes.");
    for (size_t i = 0; i < numInputNodes_; ++i) {
        auto input_node_name = session_.GetInputNameAllocated(i, allocator);
        inputNodeNameAllocatedStrings_.emplace_back(std::move(input_node_name));
        inputNames_.push_back(inputNodeNameAllocatedStrings_.back().get());
    }
    for (size_t i = 0; i < numOutputNodes_; ++i) {
        auto output_node_name = session_.GetOutputNameAllocated(i, allocator);
        outputNodeNameAllocatedStrings_.emplace_back(std::move(output_node_name));
        outputNames_.push_back(outputNodeNameAllocatedStrings_.back().get());
    }
    Ort::TypeInfo inputTypeInfo = session_.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputTensorShapeVec = inputTensorInfo.GetShape();
    if (inputTensorShapeVec.empty()) {
        throw std::runtime_error("Input tensor shape is empty. Check the model input configuration.");
    }
    if (inputTensorShapeVec.size() < 2 || inputTensorShapeVec[0] <= 0 || inputTensorShapeVec[1] <= 0) {
        throw std::runtime_error("Invalid input tensor shape: " + std::to_string(inputTensorShapeVec.size()));
    }
    isDynamicInputShape_ = (inputTensorShapeVec[2] == -1 || inputTensorShapeVec[3] == -1);
    DEBUG_PRINT("Model input tensor shape from metadata: "
                << inputTensorShapeVec[0] << "x" << inputTensorShapeVec[1] << "x"
                << inputTensorShapeVec[2] << "x" << inputTensorShapeVec[3]);
    if (!isDynamicInputShape_) {
        int modelH = static_cast<int>(inputTensorShapeVec[2]);
        int modelW = static_cast<int>(inputTensorShapeVec[3]);
        if (modelH != inputImageShape_.height || modelW != inputImageShape_.width) {
            std::cerr << "Warning: Model input shape (" << modelH << "x" << modelW
                      << ") does not match target input shape (" << inputImageShape_.height
                      << "x" << inputImageShape_.width << "). Resizing will be applied." << std::endl;
        }
    }
    if (inputTensorShapeVec.size() == 4) {
        inputImageShape_ = cv::Size(static_cast<int>(inputTensorShapeVec[3]), static_cast<int>(inputTensorShapeVec[2]));
    } else if (inputTensorShapeVec.size() == 2) {
        inputImageShape_ = cv::Size(static_cast<int>(inputTensorShapeVec[1]), static_cast<int>(inputTensorShapeVec[0]));
    } else {
        throw std::runtime_error("Unexpected input tensor shape size: " + std::to_string(inputTensorShapeVec.size()));
    }
    DEBUG_PRINT("Target input shape for preprocessing: " << inputImageShape_.height << "x" << inputImageShape_.width);
    // Load class names from the provided labels path
    classNames_ = utils::getClassNames(labelsPath);
    if (classNames_.empty()) {
        std::cerr << "Warning: Class names file is empty or failed to load. Predictions will use numeric IDs if labels are not available." << std::endl;
    } else {
        numClasses_ = static_cast<int>(classNames_.size());
        DEBUG_PRINT("Loaded " << numClasses_ << " class names from " << labelsPath);
    }
    if (numClasses_ <= 0) {
        std::cerr << "Warning: No valid number of classes determined from the model or labels file." << std::endl;
    } else {
        DEBUG_PRINT("Model predicts " << numClasses_ << " classes based on labels file.");
    }
    if (numClasses_ > 0 && !classNames_.empty() && classNames_.size() != static_cast<size_t>(numClasses_)) {
        std::cerr << "Warning: Number of classes from model (" << numClasses_
                  << ") does not match number of labels in " << labelsPath
                  << " (" << classNames_.size() << ")." << std::endl;
    }
    if (classNames_.empty() && numClasses_ > 0) {
        std::cout << "Warning: Class names file is empty or failed to load. Predictions will use numeric IDs if labels are not available." << std::endl;
    }
    if (numClasses_ <= 0 && classNames_.empty()) {
        std::cerr << "Error: No valid number of classes or class names available. Cannot perform classification." << std::endl;
        throw std::runtime_error("Invalid model configuration: No classes available.");
    }
    DEBUG_PRINT("YOLO8Classifier initialized with model: " << modelPath
                << ", labels: " << labelsPath
                << ", input shape: " << inputImageShape_.height << "x" << inputImageShape_.width);
}

void YOLO8Classifier::preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape) {
    ScopedTimer timer("Preprocessing");

    if (image.empty()) {
        std::cerr << "Error: Input image for preprocessing is empty." << std::endl;
        blob = nullptr;
        return;
    }

    cv::Mat resizedImage;
    utils::preprocessImageToTensor(image, resizedImage, inputImageShape_, cv::Scalar(0, 0, 0), true, "letterbox");

    if (resizedImage.empty()) {
        std::cerr << "Error: Preprocessed image is empty after resizing." << std::endl;
        blob = nullptr;
        return;
    }

    // Convert to float and normalize
    resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);

    // Prepare the input tensor shape
    inputTensorShape = {1, resizedImage.channels(), resizedImage.rows, resizedImage.cols};
    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);
    
    if (inputTensorSize == 0) {
        std::cerr << "Error: Input tensor size is zero after preprocessing." << std::endl;
        blob = nullptr;
        return;
    }

    // Allocate memory for the blob
    blob = new float[inputTensorSize];
    
    // Fill the blob with preprocessed data in CHW format
    int h = resizedImage.rows;
    int w = resizedImage.cols;
    int num_channels = resizedImage.channels();
    
    if (num_channels != 3) {
        delete[] blob; // Clean up allocated memory
        throw std::runtime_error("Expected 3 channels for image blob after RGB conversion, but got: " + std::to_string(num_channels));
    }
    
    for (int c_idx = 0; c_idx < num_channels; ++c_idx) {      // Iterate over R, G, B channels
        for (int i = 0; i < h; ++i) {     // Iterate over rows (height)
            for (int j = 0; j < w; ++j) { // Iterate over columns (width)
                // floatRgbImage is HWC (i is row, j is col, c_idx is channel index in Vec3f)
                // float pixel_value = resizedImage.at<cv::Vec3f>(i, j)[c_idx];
                float pixel_value = resizedImage.at<cv::Vec3f>(i, j)[c_idx];
                // Scale to [0.0, 1.0]
                float scaled_pixel = pixel_value / 255.0f;
                // Store in blob (CHW format)
                blob[c_idx * (h * w) + i * w + j] = scaled_pixel;
            }
        }
    }
    DEBUG_PRINT("Preprocessing completed (RGB, scaled [0,1]). Actual input tensor shape: "
                << inputTensorShape[0] << "x" << inputTensorShape[1] << "x"
                << inputTensorShape[2] << "x" << inputTensorShape[3]);
}

ClassificationResult YOLO8Classifier::postprocess(const std::vector<Ort::Value> &outputTensors) {
    ScopedTimer timer("Postprocessing");

    if (outputTensors.empty()) {
        std::cerr << "Error: No output tensors for postprocessing." << std::endl;
        return {};
    }

    const float* rawOutput = outputTensors[0].GetTensorData<float>();
    if (!rawOutput) {
        std::cerr << "Error: Output tensor data is null." << std::endl;
        return {};
    }

    // Assuming the output is a 1D array of scores for each class
    int currentNumClasses = static_cast<int>(outputTensors[0].GetTensorTypeAndShapeInfo().GetShape()[1]);
    if (currentNumClasses <= 0) {
        std::cerr << "Error: Invalid number of classes in output tensor." << std::endl;
        return {};
    }

    // Find the class with the highest score
    int bestClassId = -1;
    float maxScore = -std::numeric_limits<float>::infinity();
    std::vector<float> scores(rawOutput, rawOutput + currentNumClasses);

    for (int i = 0; i < currentNumClasses; ++i) {
        if (scores[i] > maxScore) {
            maxScore = scores[i];
            bestClassId = i;
        }
    }
    if (bestClassId == -1) {
        std::cerr << "Error: Could not determine best class ID." << std::endl;
        return {};
    }
    // Apply softmax to get probabilities
    float sumExp = 0.0f;
    std::vector<float> probabilities(currentNumClasses);
    // Compute softmax with numerical stability
    for (int i = 0; i < currentNumClasses; ++i) {
        probabilities[i] = std::exp(scores[i] - maxScore);
        sumExp += probabilities[i];
    }
    // Calculate final confidence
    float confidence = sumExp > 0 ? probabilities[bestClassId] / sumExp : 0.0f;
    // Get class name

    std::string className = "Unknown";
    if (bestClassId >= 0 && static_cast<size_t>(bestClassId) < classNames_.size()) {
        className = classNames_[bestClassId];
    } else if (bestClassId >= 0) {
        className = "ClassID_" + std::to_string(bestClassId);
    }
    DEBUG_PRINT("Best class ID: " << bestClassId << ", Name: " << className << ", Confidence: " << confidence);
    return ClassificationResult(bestClassId, confidence, className);
}

ClassificationResult YOLO8Classifier::classify(const cv::Mat &image) {
    ScopedTimer timer("Classification");

    if (image.empty()) {
        std::cerr << "Error: Input image for classification is empty." << std::endl;
        return {};
    }

    float *blobPtr = nullptr;
    std::vector<int64_t> currentInputTensorShape;

    try {
        preprocess(image, blobPtr, currentInputTensorShape);
    } catch (const std::exception& e) {
        std::cerr << "Exception during preprocessing: " << e.what() << std::endl;
        if (blobPtr) delete[] blobPtr;
        return {};
    }

    if (!blobPtr) {
        std::cerr << "Error: Preprocessing failed to produce a valid data blob." << std::endl;
        return {};
    }

    size_t inputTensorSize = utils::vectorProduct(currentInputTensorShape);
    if (inputTensorSize == 0) {
        std::cerr << "Error: Input tensor size is zero after preprocessing." << std::endl;
        delete[] blobPtr;
        return {};
    }

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        blobPtr, 
        inputTensorSize,
        currentInputTensorShape.data(),
        currentInputTensorShape.size()
    );

    delete[] blobPtr;
    blobPtr = nullptr;

    std::vector<Ort::Value> outputTensors;
    try {
        outputTensors = session_.Run(
            Ort::RunOptions{nullptr},
            inputNames_.data(),
            &inputTensor,
            numInputNodes_,
            outputNames_.data(),
            numOutputNodes_
        );
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Exception during Run(): " << e.what() << std::endl;
        return {};
    }

    if (outputTensors.empty()) {
        std::cerr << "Error: ONNX Runtime Run() produced no output tensors." << std::endl;
        return {};
    }

    try {
        return postprocess(outputTensors);
    } catch (const std::exception& e) {
        std::cerr << "Exception during postprocessing: " << e.what() << std::endl;
        return {};
    }
}
// End of YOLO8Classifier class implementation
// Note: The preprocess, postprocess, and classify methods remain the same as previous correct version
// End of YOLO8Classifier class definition
// End of YOLO8CLASS.hpp