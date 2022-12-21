#include "mask_rcnn.h"
#include <torch/script.h>
#include <vector>
#include "cvtoolkit.h"

DetObjs MaskRCNN::get_results(const std::vector<torch::Tensor>& datas,float r)
{
    DetObjs res;

    auto       &bbox_data   = datas[0];
    auto       &mask_data   = datas.at(1);
    const auto  mask_width  = mask_data.size(2);
    const auto  mask_height = mask_data.size(1);
    const auto  mask_size   = mask_width *mask_height;

    res.reserve(bbox_data.size(0));

    auto ndata = bbox_data.clamp(0);

    auto nr = torch::sum(ndata.sum(-1)>0).item<float>();

    for(auto i=0; i<nr; ++i) {
        if(ndata[i][4].item<float>()<score_threshold_)
            continue;
        DetObj obj;
        obj.bbox = cv::Rect(ndata[i][0].item<float>()*r,
                ndata[i][1].item<float>()*r,
                (ndata[i][2]-ndata[i][0]).item<float>()*r,
                (ndata[i][3]-ndata[i][1]).item<float>()*r);
        obj.score = ndata[i][4].item<float>();
        obj.label = ndata[i][5].item<float>();

        auto mask_ptr =  mask_data.data_ptr<uint8_t>()+mask_size*i;

        obj.mask_contours = cvt::get_mask_contours(mask_ptr,mask_width,mask_height,obj.bbox);
        res.push_back(obj);
    }
    return res;
}
