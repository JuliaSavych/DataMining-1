- Install TensorFlow: https://www.tensorflow.org/install/
    - I did the Windows Conda CPU only variant, but slim lib seems to need GPU
- Install TF-slim and TF-slim image models: https://github.com/tensorflow/models/tree/master/slim
- cd `models/slim` (or wherever you download the TF-slim image models)
- Download and convert the cifar10 dataset in the format for slim lib
    - `python download_and_convert_data.py --dataset_name=cifar10 --dataset_dir="../../cifar-10"`
    - dataset_dir is where to download the dataset
- Train a MobileNet_v1 model on the cifar10 dataset:
    - `python train_image_classifier.py --train_dir="../../train" --dataset_dir="../../cifar-10" --dataset_name=cifar10 --dataset_split_name=train --model_name=mobilenet_v1`
    - If running CPU only - add `--clone_on_cpu=True`
    - If out of ram - add `--batch_size=X` where X < 32
    - You can also modify the `depth_multiplier` on line 269 in `models/slim/nets/mobile_v1.py` to e.g. 0.25 for a smaller network
- Evaluate performance on the newly trained model over cifar10
    - `python eval_image_classifier.py --alsologtostderr --checkpoint_path="../../train/" --dataset_dir="../../cifar-10" --dataset_name=cifar10 --dataset_split_name=test --model_name=mobilenet_v1`


