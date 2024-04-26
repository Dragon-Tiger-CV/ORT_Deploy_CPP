#include "data_save.h"
#include <iostream>
#include <curl/curl.h>
#include <boost/asio.hpp>

#pragma warning(disable : 4996)

std::string get_local_ip()
{
	boost::asio::io_context io_context;
	boost::asio::ip::udp::socket socket(io_context);
	boost::asio::ip::udp::endpoint server_endpoint(boost::asio::ip::address::from_string("8.8.8.8"), 80);
	socket.connect(server_endpoint);
	boost::asio::ip::udp::endpoint local_endpoint = socket.local_endpoint();
	std::string local_ip = local_endpoint.address().to_string();
	socket.close();
	return local_ip;
}

size_t write_img(void *ptr, size_t size, size_t nmemb, void *stream)
{
	std::vector<uchar> *data = (std::vector<uchar> *)stream;
	size_t count = size * nmemb;
	data->insert(data->end(), (uchar *)ptr, (uchar *)ptr + count);
	return count;
}

std::tuple<cv::Mat, crow::json::wvalue> url_to_image(const std::string &url, crow::json::wvalue &result_info)
{
	cv::Mat img{};
	if (std::ifstream(url))
	{
		img = cv::imread(url);
	}
	else
	{
		CURL *curl;
		CURLcode res;
		std::vector<uchar> img_data;

		curl = curl_easy_init();
		if (!curl)
		{
			result_info["errnbr"] = -1;
			result_info["err_desc"] = "init url fail!";
			return std::make_tuple(img, result_info);
		}

		curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_img);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &img_data);

		res = curl_easy_perform(curl);
		if (res != CURLE_OK)
		{
			result_info["errnbr"] = -1;
			result_info["err_desc"] = "init url fail!";
			return std::make_tuple(img, result_info);
		}

		curl_easy_cleanup(curl);

		img = cv::imdecode(img_data, cv::IMREAD_COLOR);
		if (img.empty())
		{
			std::cerr << "Failed to decode image" << std::endl;
			result_info["errnbr"] = -1;
			result_info["err_desc"] = "image is empty!";
			return std::make_tuple(img, result_info);
		}
	}
	return std::make_tuple(img, result_info);
}

DataSave::DataSave(nlohmann::json config, nlohmann::json service_config)
{
	this->program_name = config["program_name"].dump();
	this->logexpiry_date = config["logexpiry_date"];
	this->imgexpiry_date = config["imgexpiry_date"];
	this->service_port = service_config["port"].dump();
	this->save_dirs_ = getsavedir();
}

std::string DataSave::data_clean(std::string data_dir, int keep_day)
{
	bf::path pdata_dir(data_dir);
	std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	std::stringstream now_day;
	now_day << std::put_time(std::localtime(&now), "%Y-%m-%d");
	std::string now_date = now_day.str();
	
	if (!bf::exists(pdata_dir)) {
		// 目录不存在，可以在这里进行创建目录的操作
		bf::create_directory(pdata_dir);
	}
	else
	{
		for (const auto &entry : bf::directory_iterator(pdata_dir))
		{
			if (!entry.is_directory())
			{
				continue;
			}
			else
			{
				std::string current_dir = entry.path().filename().string();
			
				std::tm tm1 = {};
				std::tm tm2 = {};
				std::istringstream ss1(now_date);
				std::istringstream ss2(current_dir);
				ss1 >> std::get_time(&tm1, "%Y-%m-%d");
				ss2 >> std::get_time(&tm2, "%Y-%m-%d");
				std::time_t t1 = std::mktime(&tm1);
				std::time_t t2 = std::mktime(&tm2);
				const int SECONDS_PER_DAY = 60 * 60 * 24;
				int diff_time = std::abs(std::difftime(t2, t1)) / SECONDS_PER_DAY;
				if (diff_time <= keep_day)
				{
					continue;
				}
				else
				{
					bf::remove_all(entry.path());
				}
			}
		}
	}

	bf::path psave_dir = pdata_dir / now_date;

	if (!bf::exists(psave_dir))
	{
		bf::create_directory(psave_dir);
	}

	return psave_dir.string();
}

nlohmann::json DataSave::add_datadir(std::string dirname)
{
	bf::path path = bf::current_path();
	std::string full_path = (path / "data" / this->program_name /  dirname).string();

	std::string save_dir = data_clean(full_path, this->logexpiry_date);

	this->save_dirs_[dirname] = save_dir;
	return this->save_dirs_;
}

nlohmann::json DataSave::getsavedir()
{
	bf::path fpath = bf::current_path();
	bf::path program_dir = fpath / "data" / this->program_name;
	std::string src_path = (program_dir / "src").string();
	std::string dst_path = (program_dir / "dst").string();
	std::string debug_path = (program_dir / "debug").string();
	std::string logs_path = (program_dir / "logs").string();

	std::string save_srcdir = data_clean(src_path, this->imgexpiry_date);

	std::string save_dstdir = data_clean(dst_path, this->imgexpiry_date);

	std::string save_debugdir = data_clean(debug_path, this->imgexpiry_date);

	std::string save_logsdir = data_clean(logs_path, this->logexpiry_date);
	
	nlohmann::json save_dirs;
	save_dirs = {
		{"srcdir", save_srcdir},
		{"dstdir", save_dstdir},
		{"debugdir", save_debugdir},
		{"logsdir", save_logsdir},
	};
	return save_dirs;
}

std::string DataSave::save_image(cv::Mat img, std::string img_savedir)
{
	std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	std::stringstream ss;
	ss << std::put_time(std::localtime(&now), "%Y-%m-%d_%H-%M-%S") << '_' << std::to_string(rand()) << ".jpg";
	std::string filename = ss.str();
	std::string save_path = (bf::path(img_savedir) / filename).string();
	cv::imwrite(save_path, img);
	return save_path;
}

void DataSave::write_log(std::string obj)
{
	std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	std::stringstream ss;
	ss << std::put_time(std::localtime(&now), "%Y-%m-%d_%H-%M-%S");
	std::string now_time = ss.str();
	std::ofstream file(this->save_dirs_["logsdir"].dump() + "/runlog.txt", std::ios_base::app | std::ios_base::out); 
	if (file.is_open())
	{
		if (obj == "\n")
		{
			file << obj;
		}
		else
		{
			file << now_time << ": " << obj << "\n";
		}
		file.close();
	}
}

std::string DataSave::get_urlpath(std::string local_path)
{
	std::string ip = get_local_ip();
	std::string absolutePath = bf::absolute(local_path).string();
	std::replace(absolutePath.begin(), absolutePath.end(), '\\', '/');

	std::string local_imgpath = "http://" + ip + ":" + this->service_port + "/?path=" + absolutePath;
	return local_imgpath;
}