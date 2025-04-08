<div align=center>
<h1> S&D Messenger: Exchanging Semantic and Domain
Knowledge for Generic Semi-Supervised Medical
Image Segmentation </h1>
</div>
<div align=center>

<a src="https://img.shields.io/badge/%F0%9F%93%96-ICCV_2023-red.svg?style=flat-square" href="https://arxiv.org/abs/2407.07763">
<img src="https://img.shields.io/badge/build-AcceptWithMinor-brightgreen?style=flat-square&label=TMI&color=red">
</a>

<a src="https://img.shields.io/badge/%F0%9F%9A%80-xmed_Lab-ed6c00.svg?style=flat-square" href="https://xmengli.github.io/">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-xmed_Lab-ed6c00.svg?style=flat-square">
</a>
</div>

## :rocket: Updates & Todo List

- [x] Create the repository and the ReadMe Template
- [ ] Release the training and testing codes for S&D-Messenger
- [ ] Release the pre-processed datasets (Synapse, MMWHS, LASeg, M&Ms, AMOS)
- [ ] Release the model weights for Synapse dataset (20%, 40% labeled)
- [ ] Release the model weights for MMWHS (CT2MRI, MRI2CT)
- [ ] Release the model weights for LASeg (5%, 10%)
- [ ] Release the model weights for M&Ms (Domain A, Domain B, Domain C, Domain D)

## :star: Highlights of S&D-Messenger
- S&D-Messenger attains SoTA performance simultaneously in SSMIS, UMDA, Semi-MDG
- S&D-Messenger can be seaminglessly integrated to different Transformer Blocks

## :hammer: Installation
- Main python libraries of our experimental environment are shown in [requirements.txt](./requirements.txt). You can install S&D-Messenger following:
```shell
git clone https://github.com/xmed-lab/SD-Messenger.git
cd SDMessenger
conda create -n SDMessenger
conda activate SDMessenger
pip install -r ./requirements.txt
```

## :computer: Prepare Dataset
Download the pre-processed datasets and splits from the followings:

<table>
    <tr align="center">
        <td> Dataset </td>
        <td colspan="2">Synapse Dataset</td>
        <td colspan="2">MMWHS Dataset</td>
        <td colspan="4">M&Ms Dataset</td>
        <td colspan="2">LASeg Dataset</td>
    </tr>
    <tr align="center">
        <td> Original Link </td>
        <td colspan="2">Link</td>
        <td colspan="2">Link</td>
        <td colspan="4">Link</td>
        <td colspan="2">Link</td>
    </tr>
    <tr align="center">
        <td> Pre-processed Numpy </td>
        <td colspan="2">Link</td>
        <td colspan="2">Link</td>
        <td colspan="4">Link</td>
        <td colspan="2">Link</td>
    </tr>
    <tr align="center">
        <td> Split Files </td>
        <td>20%</td>
        <td>40%</td>
        <td>CT2MRI</td>
        <td>MRI2CT</td>
        <td>Domain A</td>
        <td>Domain B</td>
        <td>Domain C</td>
        <td>Domain D</td>
		<td>5%</td>
        <td>10%</td>
    </tr>

- OOD detection datasets.
   - ID dataset, ImageNet-1K: The ImageNet-1k dataset (ILSVRC-2012) can be downloaded [here](https://image-net.org/challenges/LSVRC/2012/index.php#).
   - OOD dataset, iNaturalist, SUN, Places, and Texture. Please follow instruction from these two repositories [MOS](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) and [MCM](https://github.com/deeplearning-wisc/MCM) to download the subsampled datasets where semantically overlapped classes with ImageNet-1k are removed.

When you have downloaded the above datasets, please re-write your data root into [./src/tuning_util.py](./src/tuning_util.py).

## :key: Pre-Train and Evaluate CLIPN

- Pre-train CLIPN on CC3M. ***This step is to empower "no" logic within CLIP via the web-dataset.*** 
   - The model of CLIPN is defined in [./src/open_clip/model.py](./src/open_clip/model.py). Here, you can find a group of learnable 'no' token embeddings defined in Line 527.
   - The function of loading parameters of CLIP is defined in [./src/open_clip/factory.py](./src/open_clip/factory.py).
   - The loss functions are defined in [./src/open_clip/loss.py](./src/open_clip/loss.py).
   - You can pre-train CLIPN on ViT-B-32 and ViT-B-16 by:
```shell
cd ./src
sh run.sh
```

- Zero-Shot Evaluate CLIPN on ImageNet-1K.
   - Metrics and pipeline are defined in [./src/zero_shot_infer.py](./src/zero_shot_infer.py). Here you can find three baseline methods, and our two inference algorithms: CTW and ATD (see Line 91-96). 
   - Dataset details are defined in [./src/tuning_util.py](./src/tuning_util.py).
   - Inference models are defined in [./src/classification.py](./src/classification.py), including converting the text encoders into classifiers.
   - You can download the models provided in the table below or pre-trained by yourself. Then re-write the path of your models in the main function of [./src/zero_shot_infer.py](./src/zero_shot_infer.py). Finally, evaluate CLIPN by:
```shell
python3 zero_shot_infer.py
```


## :blue_book: Reproduced Results

***To ensure the reproducibility of the results, we conducted three repeated experiments under each configuration. The following will exhibit the most recent reproduced results achieved before open-sourcing.***

- ImageNet-1K
<table>
    <tr align="center">
        <td> Dataset </td>
        <td colspan="2">Synapse Dataset</td>
        <td colspan="2">MMWHS Dataset</td>
        <td colspan="4">M&Ms Dataset</td>
        <td colspan="2">LASeg Dataset</td>
    </tr>
    <tr align="center">
        <td> Original Link </td>
        <td colspan="2">Link</td>
        <td colspan="2">Link</td>
        <td colspan="4">Link</td>
        <td colspan="2">Link</td>
    </tr>
    <tr align="center">
        <td> Pre-processed Numpy </td>
        <td colspan="2">Link</td>
        <td colspan="2">Link</td>
        <td colspan="4">Link</td>
        <td colspan="2">Link</td>
    </tr>
    <tr align="center">
        <td> Split Files </td>
        <td>20%</td>
        <td>40%</td>
        <td>CT2MRI</td>
        <td>MRI2CT</td>
        <td>Domain A</td>
        <td>Domain B</td>
        <td>Domain C</td>
        <td>Domain D</td>
		<td>5%</td>
        <td>10%</td>
    </tr>


</table>

<font color='red'> The performance in this table is better than our paper </font>, because that we add an average learnable "no" prompt (see ***Line 600-616*** in [./src/open_clip/model.py](./src/open_clip/model.py)).

## :books: Citation

If you find our paper helps you, please kindly consider citing our paper in your publications.
```bash
@article{zhang2024s,
  title={S\&D Messenger: Exchanging Semantic and Domain Knowledge for Generic Semi-Supervised Medical Image Segmentation},
  author={Zhang, Qixiang and Wang, Haonan and Li, Xiaomeng},
  journal={arXiv preprint arXiv:2407.07763},
  year={2024}
}
```

## :beers: Acknowledge
We sincerely appreciate these three highly valuable repositories [open_clip](https://github.com/mlfoundations/open_clip), [MOS](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) and [MCM](https://github.com/deeplearning-wisc/MCM).

