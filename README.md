# AutoMolCo: <ins>Auto</ins>mated <ins>Mol</ins>ecular <ins>Co</ins>ncept Generation and Labeling with Large Language Models

This repository contains the code of the COLING 2025 paper [Automated Molecular Concept Generation and Labeling with Large Language Models](https://arxiv.org/abs/2406.09612) by Zimin Zhang, Qianli Wu, Botao Xia, Fang Sun, Ziniu Hu, Yizhou Sun, and Shichang Zhang.

## Getting Started

### Requirements

AutoMolCo is tested on Ubuntu 22.04 and MacOS Sequoia 15.2 with Python 3.9.16. Make sure to install the proper [PyTorch](https://pytorch.org/) version that is compatible with your device.

1. Clone the repository to your machine and navigate to the root directory of this repo.

```bash
git clone https://github.com/ziminz19/AutoMolCo.git
cd AutoMolCo
```

2. Create `conda` environment to manage Python package dependencies and use `pip3` to install the required packages.

```bash
conda create -n AutoMolCo python=3.9.16
conda activate AutoMolCo
pip3 install -r requirements.txt
```
