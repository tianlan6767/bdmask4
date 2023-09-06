#include "bdmapp.h"

bool BdmApp::bdminit(shared_ptr<Bdm::Infer> &bdm, const string &fcos_engine_path, const string &blender_engine_path,float mean[],float std[],int device_id)
{   

    // int device_id = 0;
    bdm = Bdm::create_infer(fcos_engine_path, blender_engine_path, device_id, 0.15, mean, std);
    // blender = Bdm::create_infer(blender_engine_path, device_id);
    if (bdm != nullptr)
    {
        return true;
    }
    else
    {
        return false;
    }
}

ObjectDetector::MaskArray BdmApp::bdmapp(shared_ptr<Bdm::Infer> bdm, cv::Mat& image){
    // FCOS推理分支
    auto masks = bdm->commit(image).get();
    return masks;
}

