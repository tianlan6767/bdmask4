#include "app_bdm/bdm.hpp"
#include "common/object_detector.hpp"

using namespace std;

class BdmApp {
public:
    bool bdminit(shared_ptr<Bdm::Infer> &bdm, 
                 const string & fcos_engine_path,  
                 const string & blender_engine_path,
                 float mean[],
                 float std[],
                 int device_id =0);
    ObjectDetector::MaskArray bdmapp(shared_ptr<Bdm::Infer> bdm, 
                                  cv::Mat& image);

};