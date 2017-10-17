# OpenNMT-py: Open-Source Neural Machine Translation

This is a [Pytorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system. It is designed to be research friendly to try out new ideas in translation, summary, image-to-text, morphology, and many other domains.


OpenNMT-py is run as a collaborative open-source project. It is currently maintained by [Sasha Rush](http://github.com/srush) (Cambridge, MA), [Ben Peters](http://github.com/bpopeters) (Saarbrücken), and [Jianyu Zhan](http://github.com/jianyuzhan) (Shenzhen). The original code was written by [Adam Lehrer](http://github.com/adamlehrer) (NYC). Codebase is nearing a stable 0.1 version. We currently recommend forking if you want stable code.

We love contributions. Please consult the Issues page for any [Contributions Welcome](https://github.com/OpenNMT/OpenNMT-py/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22) tagged post. 

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>


Table of Contents
=================

  * [Requirements](#requirements)
  * [Features](#features)
  * [Quickstart](#quickstart)
  * [Advanced](#advanced)
  * [Citation](#citation)
 
## Requirements

```bash
pip install -r requirements.txt
```


## Features

The following OpenNMT features are implemented:

- multi-layer bidirectional RNNs with attention and dropout
- data preprocessing
- saving and loading from checkpoints
- Inference (translation) with batching and beam search
- Context gate
- Multiple source and target RNN (lstm/gru) types and attention (dotprod/mlp) types
- TensorBoard/Crayon logging
- Source word features

Beta Features (committed):
- multi-GPU
- Image-to-text processing
- "Attention is all you need"
- Copy, coverage
- Structured attention
- Conv2Conv convolution model
- SRU "RNNs faster than CNN" paper
- Inference time loss functions.

## Quickstart

## Step 1: Preprocess the data

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `demo.src.dict`: Dictionary of source vocab to index mappings.
* `demo.tgt.dict`: Dictionary of target vocab to index mappings.
* `demo.train.pt`: serialized PyTorch file containing vocabulary, training and validation data


Internally the system never touches the words themselves, but uses these indices.

## Step 2: Train the model

```bash
python train.py -data data/demo -save_model demo-model
```

The main train command is quite simple. Minimally it takes a data file
and a save file.  This will run the default model, which consists of a
2-layer LSTM with 500 hidden units on both the encoder/decoder. You
can also add `-gpuid 1` to use (say) GPU 1.

## Step 3: Translate

```bash
python translate.py -model demo-model_epochX_PPL.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `pred.txt`.

!!! note "Note"
    The predictions are going to be quite terrible, as the demo dataset is small. Try running on some larger datasets! For example you can download millions of parallel sentences for [translation](http://www.statmt.org/wmt16/translation-task.html) or [summarization](https://github.com/harvardnlp/sent-summary).

## Some useful tools:


## Full Translation Example

The example below uses the Moses tokenizer (http://www.statmt.org/moses/) to prepare the data and the moses BLEU script for evaluation.

```bash
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
```

## WMT'16 Multimodal Translation: Multi30k (de-en)

An example of training for the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).

### 0) Download the data.

```bash
mkdir -p data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
wget https://staff.fnwi.uva.nl/d.elliott/wmt16/mmt16_task1_test.tgz && tar -xf mmt16_task1_test.tgz -C data/multi30k && rm mmt16_task1_test.tgz
```

### 1) Preprocess the data.

```bash
# Delete the last line of val and training files.
for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
for l in en de; do for f in data/multi30k/*.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
python preprocess.py -train_src data/multi30k/train.en.atok -train_tgt data/multi30k/train.de.atok -valid_src data/multi30k/val.en.atok -valid_tgt data/multi30k/val.de.atok -save_data data/multi30k.atok.low -lower

python preprocess.py -train_src data/multi30k/train.en.atok -train_tgt data/multi30k/train.de.atok -valid_src data/multi30k/val.en.atok -valid_tgt data/multi30k/val.de.atok -save_data data/multi30k.atok.low -lower

python preprocess.py -train_src data/multi30k/train.en.atok -train_tgt data/multi30k/train.de.atok -valid_src data/multi30k/val.en.atok -valid_tgt data/multi30k/val.de.atok -save_data data/ende.org.low -lower

python preprocess.py -train_src data/multi30k/train.de.atok -train_tgt data/multi30k/train.en.atok -valid_src data/multi30k/val.de.atok -valid_tgt data/multi30k/val.en.atok -save_data data/deen.org.low -lower

python preprocess.py -train_src data/multi30k/train.en.atok -train_tgt data/multi30k/train.de.atok -valid_src data/multi30k/val.en.atok -valid_tgt data/multi30k/val.de.atok -save_data data/ende.img.low -vgg_features_file features.mat -lower

python preprocess.py -train_src data/multi30k/train.de.atok -train_tgt data/multi30k/train.en.atok -valid_src data/multi30k/val.de.atok -valid_tgt data/multi30k/val.en.atok -save_data data/deen.img.low  -vgg_features_file features.mat-lower
```

### 2) Train the model.

```bash
python train.py -data data/multi30k.atok.low.train.pt -save_model multi30k_model -gpuid 0

python train.py -data data/ende.org.low -save_model ende_original_model -gpuid 0

python train.py -data data/deen.org.low -save_model deen_original_model -gpuid 3

python train.py -data data/ende.img.low -save_model ende_img_model -gpuid 0 -gaussian_dropout 1 -encoder_type lupi

python train.py -data data/deen.img.low -save_model deen_img_model -gpuid 3 -gaussian_dropout 1 -encoder_type lupi

```

### 3) Translate sentences.

```bash
python translate.py -gpu 0 -model ende_original_model_acc_66.12_ppl_8.46_e13.pt -src data/multi30k/test.en.atok -tgt data/multi30k/test.de.atok -replace_unk -verbose -output ende_original_model_output_e13

python translate.py -gpu 0 -model deen_original_model_acc_68.60_ppl_6.71_e13.pt -src data/multi30k/test.de.atok -tgt data/multi30k/test.en.atok -replace_unk -verbose -output deen_original_model_output_e13
```

### 4) Evaluate.

```bash
perl tools/multi-bleu.perl data/multi30k/test.de.atok < ende_original_model_output_e13

perl tools/multi-bleu.perl data/multi30k/test.en.atok < deen_original_model_output_e13
```
```
ende_original_model_output_e13: BLEU = 33.71, 65.3/40.6/27.8/19.0 (BP=0.979, ratio=0.979, hyp_len=11981, ref_len=12232)

deen_original_model_output_e13: BLEU = 38.40, 69.8/46.7/32.1/22.4 (BP=0.982, ratio=0.982, hyp_len=12825, ref_len=13058)

deen_img_model_output_e13: BLEU = 32.15, 62.5/39.5/25.7/16.8 (BP=1.000, ratio=1.083, hyp_len=14143, ref_len=13058)

ende_img_model_output_e13: BLEU = 28.10, 58.5/34.0/22.0/14.2 (BP=1.000, ratio=1.098, hyp_len=13426, ref_len=12232)

ende_1: BLEU = 38.39, 63.6/39.6/27.3/18.9 (BP=1.000, ratio=1.026, hyp_len=12546, ref_len=12232)

deen_2: BLEU = 42.34, 68.1/44.6/30.1/20.7 (BP=1.000, ratio=1.013, hyp_len=13227, ref_len=13058)

## Pretrained Models

The following pretrained models can be downloaded and used with translate.py (These were trained with an older version of the code; they will be updated soon).

- [onmt_model_en_de_200k](https://drive.google.com/file/d/0B6N7tANPyVeBWE9WazRYaUd2QTg/view?usp=sharing): An English-German translation model based on the 200k sentence dataset at [OpenNMT/IntegrationTesting](https://github.com/OpenNMT/IntegrationTesting/tree/master/data). Perplexity: 20.
- onmt_model_en_fr_b1M (coming soon): An English-French model trained on benchmark-1M. Perplexity: 4.85.


## Citation

[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
