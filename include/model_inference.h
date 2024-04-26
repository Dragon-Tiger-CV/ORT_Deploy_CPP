#ifndef INFERENCE_H
#define INFERENCE_H
// #include <vector>
// #include <map>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

bool is_nvidia_gpu_available();
bool is_intel_cpu();

class Model_Infer {
public:
    Model_Infer(nlohmann::json config);
    bool check_dynamic_input();
    std::tuple<std::vector<std::string>, std::vector<std::tuple<>>> get_input_infor();
    std::tuple<std::vector<std::string>, std::vector<std::tuple<>>> get_output_name();
    nlohmann::json get_input_feed(std::vector<cv::Mat>);
    std::vector<nlohmann::json> infer();

private:
    nlohmann::json config;
    std::string device;
    std::string task;
    std::string infer_framework;
    std::string algorithm_name;
    // onnx_session;
    int stride;
    std::string character_dict_path;
    // ocr_rec_decode

    std::vector<std::string> label_names;
    std::tuple<int> colors;

    Ort::Session session{nullptr};

    std::vector<std::string> input_names;
    std::vector<std::tuple<>> input_shapes;

    std::vector<std::string> output_names;
    std::vector<std::tuple<>> output_shapes;

    std::vector<std::tuple<>> model_inputs;
    // nms

    bool use_origimg;
    
    
};

#endif // INFERENCE_H