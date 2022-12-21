#include "faster_rcnn.h"
#include "cvtoolkit.h"

using namespace std;

FasterRCNN::FasterRCNN(const string& model_path,const cv::Size& input_size,float score_threshold,const vector<float>& mean,const vector<float>& std)
:input_size_(input_size)
,score_threshold_(score_threshold)
,mean_(mean)
,std_(std)
{
    try{
        module_ = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        cout<<"Load "<<model_path<<" faild, msg="<<e.what()<<endl;
    } catch(const std::exception& e){
        cout<<"Load "<<model_path<<" faild, msg="<<e.what()<<endl;
    }
}
FasterRCNN::~FasterRCNN()
{
}
DetObjs FasterRCNN::forward(const cv::Mat& img)
{
    float r;
    auto resized_img = cvt::resize(img,input_size_.width,input_size_.height,true,0,&r);
    at::TensorOptions opt = at::TensorOptions().dtype(torch::kUInt8);//device(at::kCUDA);
    torch::Tensor input_tensor = torch::from_blob(resized_img.data, /*sizes= */{resized_img.rows,resized_img.cols,3},opt).toType(torch::ScalarType::Float);

    auto n_input_tensor = cvt::normalize(input_tensor,{123.675, 116.28, 103.53},
    {58.395, 57.12, 57.375});

    n_input_tensor = n_input_tensor.permute({2,0,1});
    n_input_tensor = n_input_tensor.unsqueeze(0).to(at::kCUDA);

    for(auto i=0; i<n_input_tensor.sizes().size(); ++i) {
        cout<<n_input_tensor.size(i)<<endl;
    }

    std::vector<torch::jit::IValue> inputs;

    inputs.push_back(n_input_tensor);

    auto outputs = module_.forward(inputs).toTuple();

    vector<torch::Tensor> tensors;

    tensors.push_back(outputs->elements()[0].toTensor().cpu().contiguous());

    if(outputs->elements().size()>1) {
        auto tensor1 = outputs->elements()[1].toTensor().cpu().contiguous();

        if(tensor1.size(1)>0) {
            tensors.push_back(tensor1);
        }
    }

    //return get_results(tensors,vector<float>({1.0/r}))[0];
    return get_results(tensors,1.0/r);
}
torch::Tensor FasterRCNN::prepare_tensor(const cv::Mat& img,float* r)
{
    auto resized_img = cvt::resize(img,input_size_.width,input_size_.height,true,0,r);
    at::TensorOptions opt = at::TensorOptions().dtype(torch::kUInt8);//device(at::kCUDA);
    torch::Tensor input_tensor = torch::from_blob(resized_img.data, /*sizes= */{resized_img.rows,resized_img.cols,3},opt).toType(torch::ScalarType::Float);

    torch::Tensor n_input_tensor;
    if(mean_.empty()){
        n_input_tensor = input_tensor;
    } else {
        n_input_tensor = cvt::normalize(input_tensor,mean_,
            std_);
    }

    n_input_tensor = n_input_tensor.permute({2,0,1});
    n_input_tensor = n_input_tensor.to(at::kCUDA);

    return n_input_tensor;
}
vector<DetObjs> FasterRCNN::forward(const vector<cv::Mat>& imgs)
{
    vector<float> r;
    vector<torch::Tensor> tensors;
    for(auto i=0; i<imgs.size(); ++i) {
        float _r = 1.0;
        tensors.push_back(prepare_tensor(imgs[i],&_r));
        r.push_back(1.0f/_r);
    }
    auto n_input_tensor = at::stack(tensors,0);


    std::vector<torch::jit::IValue> inputs;

    inputs.push_back(n_input_tensor);

    //auto outputs = module_.forward(inputs).toTensor().cpu().contiguous();
    auto outputs = module_.forward(inputs).toTuple();

    vector<torch::Tensor> out_tensors;

    out_tensors.push_back(outputs->elements()[0].toTensor().cpu().contiguous());

    if(outputs->elements().size()>1) {
        auto tensor1 = outputs->elements()[1].toTensor().cpu().contiguous();

        if(tensor1.size(1)>0) {
            out_tensors.push_back(tensor1);
        }
    }

    
    return get_results(out_tensors,r);
}
std::vector<DetObjs> FasterRCNN::get_results(const vector<torch::Tensor>& datas,std::vector<float> r)
{
    vector<DetObjs> res;
    cout<<"Batch size: "<<datas[0].size(0)<<endl;

    auto& bbox_data = datas[0];

    for(auto i=0; i<bbox_data.size(0); ++i) {
        float _r = 1.0;
        if(r.size()>i)
            _r = r[i];
        res.push_back(get_results({bbox_data[i]},_r));
    }

    return res;
}
DetObjs FasterRCNN::get_results(const vector<torch::Tensor>& datas,float r)
{
    DetObjs res;

    auto& bbox_data = datas[0];

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
        res.push_back(obj);
    }
    return res;
}
