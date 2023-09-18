#ifndef BLENDER_HPP
#define BLENDER_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include "../common/object_detector.hpp"

namespace Blender
{

    using namespace std;
    using namespace ObjectDetector;

    typedef cv::Mat feature;

    typedef tuple<Base, std::shared_ptr<float>, float*> commit_input;


    class Infer
    {
    public:
        virtual shared_future<feature> commit(const commit_input &input) = 0;
        virtual vector<shared_future<feature>> commits(const vector<commit_input> &inputs) = 0;
    };

    shared_ptr<Infer> create_infer(const string &engine_file, int gpuid = 0);

}; // namespace BLENDER

#endif // BLENDER_HPP