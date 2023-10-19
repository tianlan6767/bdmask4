#include "app_fcos/fcos.hpp"

using namespace std;

class BdmApp {
public:
    bool bdminit(shared_ptr<Fcos::Infer> &fcos, 
                 const string & fcos_engine_path,  
                 float mean[],
                 float std[],
                 int device_id =0);
    Fcos::BoxArray bdmapp(shared_ptr<Fcos::Infer> fcos, 
                                  cv::Mat& image);

};