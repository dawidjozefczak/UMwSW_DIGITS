#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>
#include <filesystem>

struct Options {
  int image_size = 224;
  int image_size_width = 128;//112;
  int image_size_height = 128;//208;

  size_t train_batch_size = 8;
  // size_t train_batch_size = 2;
  size_t test_batch_size = 200;
  size_t iterations = 13;//10;
  size_t log_interval = 20;
  // path must end in delimiter
  std::string datasetPath = "../classes/";//"./dataset/";
  std::string infoFilePath = "../info.txt";//"info.txt";
  torch::DeviceType device = torch::kCPU;
};

static Options options;

using Data = std::vector<std::pair<std::string, long>>;

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
  using Example = torch::data::Example<>;

  Data data;

 public:
  CustomDataset(const Data& data) : data(data) {}

  Example get(size_t index) {
    std::string path = options.datasetPath + data[index].first;
    // std::cout << "Path: " << path << "\n";
    auto mat = cv::imread(path);
    assert(!mat.empty());

    cv::resize(mat, mat, cv::Size(options.image_size_width, options.image_size_height));
    // cv::imshow("mat", mat);
    // cv::waitKey(0);
    std::vector<cv::Mat> channels(3);
    cv::split(mat, channels);

    auto R = torch::from_blob(
        channels[2].ptr(),
        {options.image_size_width, options.image_size_height},
        torch::kUInt8);
    auto G = torch::from_blob(
        channels[1].ptr(),
        {options.image_size_width, options.image_size_height},
        torch::kUInt8);
    auto B = torch::from_blob(
        channels[0].ptr(),
        {options.image_size_width, options.image_size_height},
        torch::kUInt8);

    auto tdata = torch::cat({R, G, B})
                     .view({3, options.image_size_width, options.image_size_height})
                     .to(torch::kFloat);
    auto tlabel = torch::from_blob(&data[index].second, {1}, torch::kLong);
    return {tdata, tlabel};
  }

  torch::optional<size_t> size() const {
    return data.size();
  }
};

std::pair<Data, Data> readInfo() {
  Data train, test;

  std::ifstream stream(options.infoFilePath);
  assert(stream.is_open());

  long label;
  std::string path, type;

  while (true) {
    stream >> path >> label >> type;

    if (type == "train")
      train.push_back(std::make_pair(path, label));
    else if (type == "test")
      test.push_back(std::make_pair(path, label));
    else
      assert(false);

    if (stream.eof())
      break;
  }

  std::random_shuffle(train.begin(), train.end());
  std::random_shuffle(test.begin(), test.end());
  return std::make_pair(train, test);
}


// ??????????????????????????????????????????????????????//
std::pair<Data, Data> readCheck() {
  Data train_check, test_check;
  long label;
  std::string ext(".png");
  for(const auto& p : std::filesystem::directory_iterator("../check")) {
      if(p.path().extension() == ext) {
      std::cout << "Path of p: " << p.path() << "\n";
      test_check.push_back(std::make_pair(p.path(), label));// vec.push_back(p.path());     
    }
  }
  std::sort(test_check.begin(), test_check.end());
  return std::make_pair(train_check, test_check);
}

// ???????????????????????????????????????????????????


struct NetworkImpl : torch::nn::SequentialImpl {
  NetworkImpl() {
    using namespace torch::nn;
    auto stride = torch::ExpandingArray<2>({2, 2});
    torch::ExpandingArray<2> shape({-1, 32 * 32 * 24});
    push_back(Conv2d(Conv2dOptions(3, 12, 3).stride(1).padding(1)));
    push_back(Functional(torch::relu));
    push_back(Functional(torch::max_pool2d, 2, stride, 0, 1, false));
    push_back(Conv2d(Conv2dOptions(12, 24, 3).stride(1).padding(1)));
    push_back(Functional(torch::relu));
    push_back(Functional(torch::max_pool2d, 2, stride, 0, 1, false));
    push_back(Dropout2d(0.2));
    push_back(Functional(torch::reshape, shape));
    push_back(Linear(32 * 32 * 24, 10));
    // push_back(Functional(torch::nn::log_softmax, 1, torch::nullopt));
    push_back(Functional(static_cast<torch::Tensor(&)(const torch::Tensor&, int64_t, torch::optional<torch::ScalarType> )>(torch::log_softmax), 1, torch::nullopt));
    // push_back(Conv2d(Conv2dOptions(192, 256, 64)));
    // push_back(Functional(torch::relu));
    // push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));

    
    // auto stride = torch::ExpandingArray<2>({2, 2});
    // torch::ExpandingArray<2> shape({-1, 256 * 6 * 6});
    // push_back(Conv2d(Conv2dOptions(3, 64, 11).stride(4).padding(2)));
    // push_back(Functional(torch::relu));
    // push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
    // push_back(Conv2d(Conv2dOptions(64, 192, 5).padding(2)));
    // push_back(Functional(torch::relu));
    // push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
    // push_back(Conv2d(Conv2dOptions(192, 384, 3).padding(1)));
    // push_back(Functional(torch::relu));
    // push_back(Conv2d(Conv2dOptions(384, 256, 3).padding(1)));
    // push_back(Functional(torch::relu));
    // push_back(Conv2d(Conv2dOptions(256, 256, 3).padding(1)));
    // push_back(Functional(torch::relu));
    // push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
    // push_back(Functional(torch::reshape, shape));
    // push_back(Dropout());
    // push_back(Linear(256 * 6 * 6, 4096));
    // push_back(Functional(torch::relu));
    // push_back(Dropout());
    // push_back(Linear(4096, 4096));
    // push_back(Functional(torch::relu));
    // // push_back(Linear(4096, 102));
    // push_back(Linear(4096, 10));
    // push_back(Functional(torch::log_softmax, 1, torch::nullopt));
    // push_back(Functional(torch::log_softmax, 1, torch::nullopt));
  }
};
TORCH_MODULE(Network);

template <typename DataLoader>
void train(
    Network& network,
    DataLoader& loader,
    torch::optim::Optimizer& optimizer,
    size_t epoch,
    size_t data_size) {
  size_t index = 0;
  network->train();
  float Loss = 0, Acc = 0;
  
  for (auto& batch : loader) {
    auto data = batch.data.to(options.device);
    auto targets = batch.target.to(options.device).view({-1});

    auto output = network->forward(data);
    auto loss = torch::nll_loss(output, targets);
    assert(!std::isnan(loss.template item<float>()));
    auto acc = output.argmax(1).eq(targets).sum();

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    
    Loss += loss.template item<float>();
    Acc += acc.template item<float>();

    if (index++ % options.log_interval == 0) {
      auto end = std::min(data_size, (index + 1) * options.train_batch_size);

      std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
                << "\tLoss: " << Loss / end << "\tAcc: " << Acc / end
                << std::endl;
    }
  }
}

template <typename DataLoader>
void test(Network& network, DataLoader& loader, size_t data_size) {
  size_t index = 0;
  network->eval();
  torch::NoGradGuard no_grad;
  float Loss = 0, Acc = 0;

  for (const auto& batch : loader) {
    auto data = batch.data.to(options.device);
    auto targets = batch.target.to(options.device).view({-1});

    auto output = network->forward(data);
    auto loss = torch::nll_loss(output, targets);
    assert(!std::isnan(loss.template item<float>()));
    auto acc = output.argmax(1).eq(targets).sum();

    Loss += loss.template item<float>();
    Acc += acc.template item<float>();
  }

  if (index++ % options.log_interval == 0)
    std::cout << "Test Loss: " << Loss / data_size
              << "\tAcc: " << Acc / data_size << std::endl;
}

// ???????????????????????????????????????????????????????
template <typename DataLoader>
void check(Network& network, DataLoader& loader, size_t data_size, Data path_check) {
  size_t index = 0;
  size_t i = 0;
  network->eval();
  torch::NoGradGuard no_grad;
  // float Loss = 0, Acc = 0;

  for (const auto& batch : loader) {
    float Loss = 0, Acc = 0;
    std::cout << "\nImage path of check: " << path_check[i].first << "\n";
    i++;
    auto data = batch.data.to(options.device);
    auto targets = batch.target.to(options.device).view({-1});

    auto output = network->forward(data);

    // std::tuple<torch::Tensor, torch::Tensor> result = torch::max(output, 1);
    auto result = torch::max(output, 1);
  
    torch::Tensor prob = std::get<0>(result);
    torch::Tensor index = std::get<1>(result);
    auto probability = prob.item<float>();
    auto idx = index.item<long>();
    // auto probability1 = prob.accessor<float, 1>();
    // auto probability = probability1.item<float>();
    // auto idx = index.accessor<long, 1>() item<float>();


    // auto loss = torch::nll_loss(output, targets);
    // assert(!std::isnan(loss.template item<float>()));
    auto acc = output.argmax(1).eq(targets).sum();

    // Loss += loss.template item<float>();
    Acc += acc.template item<float>();

    std::cout << "Output Class: " << idx << "\tProbability: " << probability << "\tAcc: " << Acc << "\n";
  }

  // if (index++ % options.log_interval == 0)
  //   std::cout << "Test Loss: " << Loss / data_size
  //             << "\tAcc: " << Acc / data_size << std::endl;
}
// ??????????????????????????????????????????????????????????????

int main() {
  torch::manual_seed(1);

  if (torch::cuda::is_available())
    options.device = torch::kCUDA;
    // options.device = torch::kCPU;
  std::cout << "Running on: "
            << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

  auto data = readInfo();

  auto train_set =
      CustomDataset(data.first).map(torch::data::transforms::Stack<>());
  auto train_size = train_set.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_set), options.train_batch_size);

  auto test_set =
      CustomDataset(data.second).map(torch::data::transforms::Stack<>());
  auto test_size = test_set.size().value();
  auto test_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(test_set), options.test_batch_size);

  Network network;
  network->to(options.device);

  torch::optim::SGD optimizer(
      network->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));

  for (size_t i = 0; i < options.iterations; ++i) {
    train(network, *train_loader, optimizer, i + 1, train_size);
    std::cout << std::endl;
    test(network, *test_loader, test_size);
    std::cout << std::endl;
  }

  std::cout << "SAVE TO OTHER\n";
  std::string model_path = "model.pt";
  torch::serialize::OutputArchive output_archive;
  network->save(output_archive);
  output_archive.save_to(model_path);

  
  //////////////////////////////////////////////////////////
  
  int deviceID = 2;
    int apiID = cv::CAP_ANY;
    cv::VideoCapture cap;
    cv::Mat frame;
    cap.open(deviceID, apiID);
    
    if(!cap.isOpened()) {
        std::cerr << "\nCannot open video\n";
    }

    std::cout << "\nPress spacebar to terminate\n";
    
    for(;;) {
        cap.read(frame);
        if(frame.empty()) {
            std::cerr << "\nError:Blank Frame\n";
        }
        cv::Point p1(0, 300), p2(640, 300);
        cv::Point p3(0, 200), p4(640, 200);
        int thickness = 10;
        cv::line(frame, p1, p2, cv::Scalar(255, 0, 0), thickness, cv::LINE_8);
        cv::line(frame, p3, p4, cv::Scalar(255, 0, 0), thickness, cv::LINE_8);
        cv::imshow("video", frame);

        char key = cv::waitKey(1);

        if(key == ' ') {
            cv::Mat croped = frame(cv::Range(200, 300), cv::Range(0, 640));
            cv::imshow("croped", croped);
            for(int i = 0; i < 7; i++) {
                std::ostringstream name;
                name << "../check1/" << i << ".png";
                cv::Mat number = croped(cv::Range(0, croped.rows), cv::Range(i*91, i*91+91));
                cv::imwrite(name.str(), number);
            }
            auto data_check = readCheck();
            auto check_set = CustomDataset(data_check.second).map(torch::data::transforms::Stack<>());
            auto check_size = check_set.size().value();
            auto check_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(check_set), 1);
            auto check_path = data_check.second;
            check(network, *check_loader, check_size, check_path);
        }
        else if (key == 'q' || key == 'Q') {
            break;
        }        
    }
  
  
  
  // auto data_check = readCheck();
  // auto check_set = CustomDataset(data_check.second).map(torch::data::transforms::Stack<>());
  // auto check_size = check_set.size().value();
  // auto check_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(check_set), 1);
  // auto check_path = data_check.second;
  // check(network, *check_loader, check_size, check_path);

  std::cout << "END\n";


  return 0;
}
