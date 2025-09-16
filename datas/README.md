# Datasets

We utilize the [MindMerger](https://github.com/CONE-MT/MindMerger) dataset for our experiments. You can download the dataset [here](https://drive.google.com/drive/folders/1Rm5ppr1fCd4KbiDR2LSFKNChq_uSfiSE?usp=drive_link) and place it in the current directory.

It is important to note that MindMerger later modified its mathematical training dataset. Our study is based on the initial version of the mathematical dataset. Since this version is no longer available in the official documentation, we have provided access to it [here](https://drive.google.com/drive/folders/1evjD7HMLPBel1GKXtg-z77dR8DuCquPl?dmr=1&ec=wgc-drive-hero-goto).


# Translation data

You can download the dataset [bilingual_pairs](https://drive.google.com/drive/folders/1Rm5ppr1fCd4KbiDR2LSFKNChq_uSfiSE?usp=drive_link) and place it in the datas/bilingual_pairs


# Math data
You can download the dataset [metamath_615k.json](https://drive.google.com/drive/folders/1evjD7HMLPBel1GKXtg-z77dR8DuCquPl?dmr=1&ec=wgc-drive-hero-goto) and place it in the datas/query_translation/metamath_615k.json

# Evalution data
You can download the dataset [mgsm](https://drive.google.com/drive/folders/1Rm5ppr1fCd4KbiDR2LSFKNChq_uSfiSE?usp=drive_link) and place it in the ./datas/evaluation/mgsm

# MindMerger data
To use the same data as MindMerger run this
```
wget https://datasets-and-checkpoints.s3.us-east-1.amazonaws.com/MindMerger-001.zip
unzip MindMerger-001.zip
mv -f MindMerger/datas/* ./datas/
```