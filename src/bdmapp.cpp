#include "bdmapp.h"

bool BdmApp::bdminit(shared_ptr<Fcos::Infer> &fcos, shared_ptr<Blender::Infer> &blender, const string &fcos_engine_path, const string &blender_engine_path,float mean[],float std[],int device_id)
{   

    // int device_id = 0;
    fcos = Fcos::create_infer(fcos_engine_path, device_id, 0.09, mean, std);
    // blender = Blender::create_infer(blender_engine_path, device_id);
    if (fcos != nullptr)
    {
        return true;
    }
    else
    {
        return false;
    }
}

Fcos::BoxArray BdmApp::bdmapp(shared_ptr<Fcos::Infer> fcos, shared_ptr<Blender::Infer>  blender, cv::Mat& image){
    // FCOS推理分支
    auto boxes = fcos->commit(image).get();
    // auto  fcos_base = boxes.BasesArray[0];
    // vector<std::tuple<ObjectDetector::Base, std::shared_ptr<float>, float*>> commit_inputs;
    // ObjectDetector::Defect defect_obj;
    // for(auto& box : boxes){
    //     std::shared_ptr<float> boxin(new float[5]);
    //     boxin.get()[0] = 0;
    //     boxin.get()[1] = box.left;
    //     boxin.get()[2] = box.top;
    //     boxin.get()[3] = box.right;
    //     boxin.get()[4] = box.bottom;
    //     commit_inputs.emplace_back(std::make_tuple(fcos_base, 
    //                                                 move(boxin), 
    //                                                 box.top_feat
    //                                                 ));
    //     defect_obj.labels.emplace_back(box.class_label);
    //     defect_obj.scores.emplace_back(sqrt(box.confidence));  
    //     defect_obj.boxes.emplace_back(box);                               
    // }
    // if (commit_inputs.size() > 0){
    //     auto masks = blender->commits(commit_inputs);
    //     for(auto& m : masks){
    //         auto img_mask = m.get();
    //         defect_obj.masks.emplace_back(img_mask);
    //     }
    // }
    return boxes;
}
