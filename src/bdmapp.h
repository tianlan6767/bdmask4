#include "app_fcos/fcos.hpp"
#include "app_blender/blender.hpp"
#include "common/object_detector.hpp"

using namespace std;

class BdmApp {
public:
    bool bdminit(shared_ptr<Fcos::Infer> &fcos, 
                 shared_ptr<Blender::Infer> &blender, 
                 const string & fcos_engine_path,  
                 const string & blender_engine_path,
                 float mean[],
                 float std[],
                 int device_id =0);
    ObjectDetector::Defect bdmapp(shared_ptr<Fcos::Infer> fcos, 
                                  shared_ptr<Blender::Infer> blender,
                                  cv::Mat& image);

};