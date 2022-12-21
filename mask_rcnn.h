#include "faster_rcnn.h"
class MaskRCNN:public FasterRCNN
{
    public:
        using FasterRCNN::FasterRCNN;
    protected:
        virtual DetObjs get_results(const std::vector<torch::Tensor>& data,float r=1.0f)override;
};
