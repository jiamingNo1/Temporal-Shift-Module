## Hand Gesture Recognition Online Demo with TSM

### Prerequisites
* tvm
```
sudo apt install llvm
git clone https://github.com/dmlc/tvm.git
cd tvm
git submodule update --init
mkdir build
cp cmake/config.cmake build/
cd build
#[
#edit config.cmake to change
# 32 line: USE_CUDA OFF -> USE_CUDA ON
#104 line: USE_LLVM OFF -> USE_LLVM ON
#]
cmake ..
make -j8
cd ..
cd python; sudo python3 setup.py install; cd ..
cd nnvm/python; sudo python3 setup.py install; cd ../..
cd topi/python; sudo python3 setup.py install; cd ../..
```
* onnx
```
sudo apt-get install protobuf-compiler libprotoc-dev
pip3 install onnx
```
* add cuda path

`export PATH=$PATH:/usr/local/cuda/bin`

### Run The Demo
Firstly, export the pytorch model. 

`cp xxx/xxx.pt.tar ./mobilenetv2_jester.pth.tar`

Then, run the demo. The first run will compile pytorch model into onnx model, 
and then compile the onnx model into tvm binary, finally run it. 
Later run will directly execute the compiled tvm model.

`python3 main.py`

Press `Q` or `Esc` to quit. Press `F` to enter or exit full screen.