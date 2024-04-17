#include "bdmapp.h"

bool BdmApp::bdminit(shared_ptr<Fcos::Infer> &fcos, const string &fcos_engine_path,float mean[],float std[],int device_id)
{   

    fcos = Fcos::create_infer(fcos_engine_path, device_id, 0.09, mean, std);
    if (fcos != nullptr)
    {
        return true;
    }
    else
    {
        return false;
    }
}

Fcos::BoxArray BdmApp::bdmapp(shared_ptr<Fcos::Infer> fcos, cv::Mat& image){
    // FCOS推理分支
    auto boxes = fcos->commit(image).get();
    return boxes;
}
