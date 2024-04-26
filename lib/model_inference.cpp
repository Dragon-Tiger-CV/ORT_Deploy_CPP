#include <iostream>
#include <string>
#include <intrin.h>
#include <boost/process.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <model_inference.h>

namespace bf = boost::filesystem;

bool is_nvidia_gpu_available()
{
    std::string command;
    #ifdef _WIN32
        command = "wmic path win32_VideoController get name";
    #else
        command = "lspci | grep -i nvidia";
    #endif

    boost::process::ipstream output;
    boost::process::system(command, boost::process::std_out > output);

    std::string line;
    bool nvidiaFound = false;
    while (std::getline(output, line))
    {
        if (line.find("nvidia") != std::string::npos)
        {
            nvidiaFound = true;
            break;
        }
    }
    return nvidiaFound;
}

bool is_intel_cpu() {
    std::array<int, 4> cpui;
    std::vector<std::array<int, 4>> data_;
    __cpuid(cpui.data(), 0);
    int nIds_ = cpui[0];

    for (int i = 0; i <= nIds_; ++i)
    {
        __cpuidex(cpui.data(), i, 0);
        data_.push_back(cpui);
    }
    char vendor[0x20];
    memset(vendor, 0, sizeof(vendor));
    *reinterpret_cast<int*>(vendor) = data_[0][1];
    *reinterpret_cast<int*>(vendor + 4) = data_[0][3];
    *reinterpret_cast<int*>(vendor + 8) = data_[0][2];

    std::string infor = vendor;
    return infor == "GenuineIntel";
}

Model_Infer::Model_Infer(nlohmann::json config)
{
    this->config = config;
    std::string weight_path = config["weight"].dump();
    bf::path root_path = bf::current_path();

    std::wstring onnx_path;
    if (bf::exists(weight_path))
    {
        onnx_path = bf::path(weight_path).wstring();
    }
    else
    {
        onnx_path = (root_path / "weights" / weight_path).wstring();
    }

    this->device = config.value("device", "cpu");
    this->task = config.value("task", "detect");
    this->infer_framework = config.value("infer_framework", "onnxruntime");

    if (this->device != "cpu" && !is_nvidia_gpu_available())
    {
        this->device = "cpu";
    }

    // #ifdef _WIN32
    //     std::wstring w_modelPath = utils::charToWstring(modelPath.c_str());
    // #else
    //     session = Ort::Session(env, modelPath.c_str(), sessionOptions);
    // #endif

    if (this->device == "cpu")
    {
        // 判断是否使用openvino推理
        // if (is_intel_cpu && this->infer_framework == "openvino"){

        // }
        // else:
        Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_INFERENCE");
        
        Ort::SessionOptions sessionOptions = Ort::SessionOptions();

        this->session = Ort::Session(env, onnx_path.c_str(), sessionOptions);

    }
    else if (this->infer_framework == "onnxruntime"){

        Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_INFERENCE");
        
        Ort::SessionOptions sessionOptions = Ort::SessionOptions();

        std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
        auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
        OrtCUDAProviderOptions cudaOption;
        if (cudaAvailable == availableProviders.end())
        {
            std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
            std::cout << "Inference device: CPU" << std::endl;
        }
        else
        {
            std::cout << "Inference device: GPU" << std::endl;
            cudaOption.device_id = std::stoi(device);
            sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
        }

        this->session = Ort::Session(env, onnx_path.c_str(), sessionOptions);
    }

    this->stride = (this->task == "ocr_rec" || this->task == "semantic_segment") ? 1 : 32;
    auto class_names = config.value("names", nlohmann::json{});

    if (this->task == "orc_rec"){
        this->character_dict_path = config.value("keys_path", nlohmann::json{});
    }
    
    if (this->infer_framework == "onnxruntime"){
        Ort::ModelMetadata metadata = session.GetModelMetadata();
        Ort::AllocatorWithDefaultOptions allocator;
        auto names = metadata.LookupCustomMetadataMapAllocated("names", allocator);
        if (names == nullptr){
            // TODO: 从配置文件读入
        }
        else{
            boost::regex pattern("'(.*?)'"); // 匹配单引号内容的正则表达式模式
            boost::sregex_token_iterator iter(std::string(names.get()).begin(), std::string(names.get()).end(), pattern, 1);
            boost::sregex_token_iterator end;

            while (iter != end)
            {
                this->label_names.push_back(*iter);
                ++iter;
            }
        }

        auto stride = metadata.LookupCustomMetadataMapAllocated("stride", allocator);
        if (stride == nullptr){
            // TODO: 从配置文件读入
        }
        else{
            this->stride = std::stoi(stride.get());
        }
        
    }
}

bool Model_Infer::check_dynamic_input()
{
}

std::tuple<std::vector<std::string>, std::vector<std::tuple<>>> Model_Infer::get_input_infor()
{
}

std::tuple<std::vector<std::string>, std::vector<std::tuple<>>> Model_Infer::get_output_name()
{
}

nlohmann::json Model_Infer::get_input_feed(std::vector<cv::Mat>)
{
}

std::vector<nlohmann::json> Model_Infer::infer()
{
}
