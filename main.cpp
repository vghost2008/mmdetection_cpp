#include "faster_rcnn.h"
#include <iostream>
#include "cvtoolkit.h"

using namespace std;
namespace vision {
  int64_t cuda_version();
}

int main(int argc,char** argv)
{
    if(argc<3) {
        cout<<"Usage: ./run model_path img_path"<<endl;
        return -1;
    }
    cout<<vision::cuda_version()<<endl;

    auto model = FasterRCNN(argv[1],cv::Size(1216,800),0.2);
    auto img = cv::imread(argv[2]);
    auto objs = model.forward(img.clone());

    cvt::draw_bboxes_dets(img,objs);
    cv::imwrite("outputs.jpg",img);

    return 0;
}
