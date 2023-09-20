# Viewpoint-Adaptive Representation Disentanglement for Change Captioning

This package contains the accompanying code for the following paper:

Tu, Yunbin, et al. ["Viewpoint-Adaptive Representation Disentanglement for Change Captioning."](https://ieeexplore.ieee.org/document/10108947), which has appeared as a regular paper in IEEE TIP 2023. 

## Installation
1. Clone this repository
2. cd VARD
1. Make virtual environment with Python 3.5 
2. Install requirements (`pip install -r requirements.txt`)
3. Setup COCO caption eval tools ([github](https://github.com/mtanti/coco-caption)) 
4. A TITAN Xp GPU or others.

## Data
1. Download data from here: [google drive link](https://drive.google.com/file/d/1HJ3gWjaUJykEckyb2M0MB4HnrJSihjVe/view?usp=sharing)
```
python google_drive.py 1HJ3gWjaUJykEckyb2M0MB4HnrJSihjVe clevr_change.tar.gz
tar -xzvf clevr_change.tar.gz
```
Extracting this file will create `data` directory and fill it up with CLEVR-Change dataset.

2. Preprocess data

The preprocessed data here: [google drive link](https://drive.google.com/file/d/1FA9mYGIoQ_DvprP6rtdEve921UXewSGF/view?usp=sharing).
You can skip the procedures explained below and just download them using the following command:
```
python google_drive.py 1FA9mYGIoQ_DvprP6rtdEve921UXewSGF ./data/clevr_change_features.tar.gz
cd data
tar -xzvf clevr_change_features.tar.gz
```

* Extract visual features using ImageNet pretrained ResNet-101:
```
# processing default images
python scripts/extract_features.py --input_image_dir ./data/images --output_dir ./data/features --batch_size 128

# processing semantically changes images
python scripts/extract_features.py --input_image_dir ./data/sc_images --output_dir ./data/sc_features --batch_size 128

# processing distractor images
python scripts/extract_features.py --input_image_dir ./data/nsc_images --output_dir ./data/nsc_features --batch_size 128
```

* Build vocab and label files of VARD-LSTM by using caption annotations:
```
python scripts/preprocess_captions.py --input_captions_json ./data/change_captions.json --input_neg_captions_json ./data/no_change_captions.json --input_image_dir ./data/images --split_json ./data/splits.json --output_vocab_json ./data/vocab.json --output_h5 ./data/labels.h5
```

## Training
To train the proposed method, run the following commands:
```
# create a directory or a symlink to save the experiments logs/snapshots etc.
mkdir experiments
# OR
ln -s $PATH_TO_DIR$ experiments

# this will start the visdom server for logging
# start the server on a tmux session since the server needs to be up during training
python -m visdom.server

# start training for VARD-LSTM
python train.py --cfg configs/dynamic/VARD.yaml  --entropy_weight 0.0001
```
## Testing/Inference
To test/run inference on the test dataset, run the following command
```
python test.py --cfg configs/dynamic/VARD.yaml --visualize --snapshot 8000 --gpu 1
```
The command above will take the model snapshot at 8000th iteration and run inference using GPU ID 1, saving visualizations as well.

## Evaluation
* Caption evaluation

To evaluate captions, we need to first reformat the caption annotations into COCO eval tool format (only need to run this once). After setting up the COCO caption eval tools ([github](https://github.com/tylin/coco-caption)), make sure to modify `utils/eval_utils.py` so that the `COCO_PATH` variable points to the COCO eval tool repository. Then, run the following command:
```
python utils/eval_utils.py
```

After the format is ready, run the following command to run evaluation:
```
# This will run evaluation on the results generated from the validation set and print the best results
python evaluate.py --results_dir ./experiments/VARD_LSTM/eval_sents --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
```

Once the best model is found on the validation set, you can run inference on test set for that specific model using the command exlpained in the `Testing/Inference` section and then finally evaluate on test set:
```
python evaluate.py --results_dir ./experiments/VARD_LSTM/test_output/captions --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
```
The results are saved in `./experiments/VARD_LSTM/test_output/captions/eval_results.txt`

If you find this helps your research, please consider citing:
```
@ARTICLE{tu2023viewpoint,
  author={Tu, Yunbin and Li, Liang and Su, Li and Du, Junping and Lu, Ke and Huang, Qingming},
  journal={IEEE Transactions on Image Processing}, 
  title={Viewpoint-Adaptive Representation Disentanglement Network for Change Captioning}, 
  year={2023},
  volume={32},
  pages={2620-2635},
  doi={10.1109/TIP.2023.3268004}}
```

## Contact
My email is tuyunbin1995@foxmail.com.


