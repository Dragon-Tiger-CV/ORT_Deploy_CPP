#ifndef DATASAVE_H
#define DATASAVE_H

#include <string>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>
#include "crow.h"

namespace bf = boost::filesystem;

std::string get_local_ip();

std::tuple<cv::Mat, crow::json::wvalue> url_to_image(const std::string& url, crow::json::wvalue& result_info);

class DataSave {
public:
    DataSave(nlohmann::json config, nlohmann::json service_config);
    std::string data_clean(std::string data_dir, int keep_day = 3);
    nlohmann::json add_datadir(std::string dirname);
    nlohmann::json getsavedir();
    std::string save_image(cv::Mat img, std::string img_savedir);
    void write_log(std::string obj);
    std::string get_urlpath(std::string local_path);

private:
    std::string program_name;
    int logexpiry_date;
    int imgexpiry_date;
    std::string service_port;
    nlohmann::json save_dirs_;
};

#endif // DATASAVE_H