#include "faster_rcnn.h"
#include "mask_rcnn.h"
#include <iostream>
#include "cvtoolkit.h"

using namespace std;
namespace vision {
  int64_t cuda_version();
}

cv::Mat prepare_img(const string& path)
{
    cout<<"Read "<<path<<endl;
    auto img = cv::imread(path);

    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);

    return img.clone();
}

int main(int argc,char** argv)
{
    if(argc<3) {
        cout<<"Usage: ./run model_path imgs_path"<<endl;
        return -1;
    }
    cout<<vision::cuda_version()<<endl;

    //auto model = FasterRCNN(argv[1],cv::Size(1216,800),0.2);
    auto model = MaskRCNN(argv[1],cv::Size(1024,640),0.5);

    /*vector<cv::Mat> imgs;

    for(auto i=2; i<argc; ++i) {
        imgs.push_back(prepare_img(argv[i]));
    }

    auto objs = model.forward(imgs);

    for(auto i=0; i<objs.size(); ++i) {
        auto cur_img = imgs[i].clone();
        cv::cvtColor(cur_img,cur_img,cv::COLOR_RGB2BGR);
        //cvt::draw_bboxes_dets(cur_img,objs[i]);
        cvt::draw_masks_dets(cur_img,objs[i]);
        auto save_path = string("outputs")+std::to_string(i)+".jpg";
        cout<<"Save path "<<save_path<<endl;
        cv::imwrite(save_path,cur_img);
    }*/
    for(auto i=2; i<argc; ++i) {
        auto img = prepare_img(argv[i]);
        auto cur_img = img.clone();

        auto objs = model.forward(img);

        cv::cvtColor(cur_img,cur_img,cv::COLOR_RGB2BGR);
        //cvt::draw_bboxes_dets(cur_img,objs[i]);
        cout<<"objs nr "<<objs.size()<<endl;
        cvt::draw_masks_dets(cur_img,objs);
        auto save_path = string("outputs")+std::to_string(i)+".jpg";
        cout<<"Save path "<<save_path<<endl;
        cv::imwrite(save_path,cur_img);
    }

    return 0;
}
