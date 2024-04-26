#include <iostream>
#include <data_save.h>
// #include <model_inference.h>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
// #include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
// #include <onnxruntime/core/providers/cpu/cpu_execution_provider.h>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
using namespace std;
namespace bf = boost::filesystem;

int main(int argc, char **argv)
{
    string ip = get_local_ip();
    cout << ip << endl;
    // bool is_gpu = is_nvidia_gpu_available();
    // cout << is_gpu << endl;
    // bool is_intel = is_intel_cpu();
    // cout << is_intel << endl;

    //    string input="{0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6:'six', 7:'seven', 8: 'eight', 9: 'nine', 10: 'burden'}";

    string device = "cpu";

    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;
    if ((device != "cpu") && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if ((device != "cpu") && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }
    bf::path onnx_path = "D:/python_codes/inspection_algorithm/weights/digital_num_ddh.onnx";
    Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_INFERENCE");
    Ort::Session session = Ort::Session(env, onnx_path.wstring().c_str(), sessionOptions);

    
    Ort::ModelMetadata metadata = session.GetModelMetadata();
    Ort::AllocatorWithDefaultOptions allocator;
    try{
        auto name = metadata.LookupCustomMetadataMapAllocated("name", allocator);
        if (name == nullptr){
            std::cout << "name is null" << std::endl;
        }
        else{
            std::cout << "name is " << name << std::endl;
        }

        std::string names = metadata.LookupCustomMetadataMapAllocated("names", allocator).get();
        std::vector<std::string> extractedStrings;
        boost::regex pattern("'(.*?)'"); // 匹配单引号内容的正则表达式模式
        boost::sregex_token_iterator iter(names.begin(), names.end(), pattern, 1);
        boost::sregex_token_iterator end;

        while (iter != end)
        {
            extractedStrings.push_back(*iter);
            ++iter;
        }
        // std::cout << extractedStrings[0] << std::endl;
        // 打印提取的内容
        for (string str : extractedStrings)
        {
            std::cout << str << std::endl;
        }
    }
    catch (runtime_error) {
        std::cout << "Exception caught: " << std::endl;
        return 0;
    }

    

    int stride = std::stoi(metadata.LookupCustomMetadataMapAllocated("stride", allocator).get());
    //  转为int
    std::cout << stride << std::endl;
    // cout << names << endl;

    return 0;
}
