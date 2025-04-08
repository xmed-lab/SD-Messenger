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
:hammer_and_wrench: There are still many TODOs, we will update the final version ASAP.
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
cd SD_Messenger
conda create -n SDMessenger
conda activate SDMessenger
pip install -r ./requirements.txt
```

## :computer: Prepare Dataset

Download the pre-processed datasets and splits from the following links:

<table>
    <tr align="center">
        <td>Dataset</td>
        <td colspan="2">Synapse Dataset</td>
        <td colspan="2">MMWHS Dataset</td>
        <td colspan="4">M&Ms Dataset</a></td>
        <td colspan="2">LASeg Dataset</td>
    </tr>
    <tr align="center">
        <td>Paper Link</td>
        <td colspan="2"><a href="https://arxiv.org/pdf/2309.03493">Link</a></td>
        <td colspan="2"><a href="https://www.sciencedirect.com/science/article/pii/S1361841516000219">Link</a></td>
        <td colspan="4"><a href="https://ieeexplore.ieee.org/document/9458279">Link</a></td>
        <td colspan="2"><a href="https://arxiv.org/abs/2004.12314">Link</a></td>
    </tr>
    <tr align="center">
        <td>Pre-processed Numpy</td>
        <td colspan="2"><a href="#">Link</a></td>
        <td colspan="2"><a href="#">Link</a></td>
        <td colspan="4"><a href="#">Link</a></td>
        <td colspan="2"><a href="#">Link</a></td>
    </tr>
    <tr align="center">
        <td>Split Files</td>
        <td><a href="#">20%</a></td>
        <td><a href="#">40%</a></td>
        <td><a href="#">CT2MRI</a></td>
        <td><a href="#">MRI2CT</a></td>
        <td><a href="#">Domain A</a></td>
        <td><a href="#">Domain B</a></td>
        <td><a href="#">Domain C</a></td>
        <td><a href="#">Domain D</a></td>
        <td><a href="#">5%</a></td>
        <td><a href="#">10%</a></td>
    </tr>
</table>


- Place the split files in ./splits
- Specify the dataset root path in the config files

## :key: Train and Evaluate S&D-Messenger

- Train S&D-Messenger on Synapse, LASeg, MMWHS, M&Ms datasets:
   - Place the dataset and split files in the corresponding folder path
   - The model of S&D-Messenger is defined in [./SD_Messenger/model](./src/model), with a backbone, and a SDMessengerTransformer
   - You can train S&D-Messenger on different datasets
```shell
cd ./SD_Messenger

# Synapse Dataset
bash scripts/train_synapse.sh ${GPU_NUM} ${PORT}
# LASeg Dataset
bash scripts/train_laseg.sh ${GPU_NUM} ${PORT}
# MMWHS Dataset
bash scripts/train_mmwhs.sh ${GPU_NUM} ${PORT}
# M&Ms Dataset
bash scripts/train_mm.sh ${GPU_NUM} ${PORT}
```

- Train S&D-Messenger on Synapse, LASeg, MMWHS, M&Ms datasets:
```shell
# Synapse Dataset
bash scripts/test_synapse.sh ${GPU_NUM} ${PORT}
# LASeg Dataset
bash scripts/test_laseg.sh ${GPU_NUM} ${PORT}
# MMWHS Dataset
bash scripts/test_mmwhs.sh ${GPU_NUM} ${PORT}
# M&Ms Dataset
bash scripts/test_mm.sh ${GPU_NUM} ${PORT}
```


## :blue_book: Reproduced Results

***To improve the reproducibility of the results, we have shared the trained model weights and training logs for your reference.***

<table>
    <tr align="center">
        <td> Dataset </td>
        <td colspan="2">Synapse Dataset</td>
        <td colspan="2">MMWHS Dataset</td>
        <td colspan="4">M&Ms Dataset</td>
        <td colspan="2">LASeg Dataset</td>
    </tr>
    <tr align="center">
        <td> Splits </td>
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
    <tr align="center">
        <td> DICE (%)</td>
        <td>64.46</td>
        <td>68.07</td>
        <td>77.0</td>
        <td>91.4</td>
        <td>86.50</td>
        <td>87.58</td>
        <td>88.30</td>
        <td>89.76</td>
				<td>90.21</td>
        <td>91.46</td>
    </tr>
   <tr align="center">
        <td> Model Weights </td>
        <td><a href="#">weight</a></td>
        <td><a href="#">weight</a></td>
        <td><a href="#">weight</a></td>
        <td><a href="#">weight</a></td>
        <td><a href="#">weight</a></td>
        <td><a href="#">weight</a></td>
        <td><a href="#">weight</a></td>
        <td><a href="#">weight</a></td>
				<td><a href="#">weight</a></td>
        <td><a href="#">weight</a></td>
    </tr>
  <tr align="center">
        <td> Training Logs </td>
        <td><a href="#">log</a></td>
        <td><a href="#">log</a></td>
        <td><a href="#">log</a></td>
        <td><a href="#">log</a></td>
        <td><a href="#">log</a></td>
        <td><a href="#">log</a></td>
        <td><a href="#">log</a></td>
        <td><a href="#">log</a></td>
				<td><a href="#">log</a></td>
        <td><a href="#">log</a></td>
    </tr>

</table>

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

We sincerely appreciate these three highly valuable repositories [UniMatch](https://github.com/LiheYoung/UniMatch), [SegFormer](https://github.com/NVlabs/SegFormer) and [A&D](https://github.com/xmed-lab/GenericSSL).

