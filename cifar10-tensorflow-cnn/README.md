To install tensorflow on Windows:
- Install python 3.5.*, 64bit version
- Make sure you have the latest version of pip:<br/>
  _$pip install --upgrade pip_<br/>
- Install tensorflow<br/>
  _$pip install --upgrade tensorflow_<br/>
  If you want to run it on the gpu, install tensorflow-gpu:<br/>
  http://www.nvidia.com/object/gpu-accelerated-applications-tensorflow-installation.html
  https://www.tensorflow.org/tutorials/using_gpu

You can run the training on CPU, GPU or multiple GPUs:<br/>
- Depending on which version of tensorflow you have currently active<br/>
  _$python cifar10_train.py_<br/>
  will run on the CPU or the GPU
- To run on multiple(or one) GPUs:<br/>
  _$python cifar10_multi_gpu_train.py_
  
To start the evaluation:<br/>
_$python cifar10_eval.py_

On Windows, the results are outputted at C:\tmp folder.<br/>
The training and the evaluation should be started simultaneously. This way, you will be able to see the progress in TensorBoard:<br/>
_$tensorboard --logdir=C:\tmp\cifar10_eval_

The training takes about 5 hours to achieve 85-86% precision on CPU.<br/>

Here you can find already trained data with 85.8% precision:<br/>
https://drive.google.com/open?id=0B7vEGqMP_g6wSVp1VVhVS3pEX1k<br/>
just extract it at C:\tmp\cifar10_train

And here you can find the evaluation of that training:<br/>
https://drive.google.com/open?id=0B7vEGqMP_g6wS0ZnZkZqZUNtUGc<br/>
just extract it at C:\tmp\cifat10_eval
