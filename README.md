#mmdetection faster-rcnn example

Model in this project use roi_align, nms ops in torchvision

## Usage

```
mkdir build
cd build
make -j4
./faster_rcnn ../faster_rcnn.traced ../demo.jpg
```

Example model can be download from [here](https://pan.baidu.com/s/1duWvkpu9-lpMyu84XsHnNg?pwd=4xsq)



## requirements
- libtorch
- [torchvision_ops](https://github.com/vghost2008/torchvision_ops)

