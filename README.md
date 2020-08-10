# Learning to Segment Actions from Observation and Narration  

Code for the paper:  
[Learning to Segment Actions from Observation and Narration](https://arxiv.org/abs/2005.03684)  
Daniel Fried, Jean-Baptiste Alayrac, Phil Blunsom, Chris Dyer, Stephen Clark, and Aida Nematzadeh  
ACL, 2020

## Summary

This repository provides a system for segmenting and labeling actions in a video, using a simple generative segmental (hidden semi-Markov) model of the video. This model can be used as a strong baseline for action segmentation on instructional video datasets such as [CrossTask](https://github.com/DmZhukov/CrossTask) ([Zhukov et al., CVPR 2019](https://arxiv.org/abs/1903.08225)), and can be trained fully supervised (with action labels for each frame in each video) or with weak supervision from narrative descriptions and "canonical" step orderings. Please see our paper for more details.

## Requirements

* pytorch 1.2
* The `semimarkov` branch of my fork of [pytorch-struct](https://github.com/dpfried/pytorch-struct/tree/semimarkov). (Newer versions may run out of memory on the long videos in the CrossTask dataset, due to changes to pytorch-struct that improve runtime complexity but increase memory usage.) It can be installed via

```bash
pip install git+https://github.com/dpfried/pytorch-struct.git@semimarkov
```

See `env.yml` for a full list of other dependencies, which can be installed with conda.

## Setup

1. Download and unpack the CrossTask dataset of Zhukov et al.:

```bash
cd data
mkdir crosstask
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_features.zip
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_constraints.zip
unzip '*.zip'
```

2. Preprocess the features with PCA. In the repository's root folder, run

```bash
PYTHONPATH="src/":$PYTHONPATH python src/data/crosstask.py
```

This should generate the folder `data/crosstask/crosstask_processed/crosstask_primary_pca-200_with-bkg_by-task`

3. To run our supervised system, run

```bash
./run_crosstask_i3d-resnet-audio.sh pca_semimarkov_sup --classifier semimarkov --training supervised
```

## Experiments

Here are the commands to replicate key results in our paper. Please contact Daniel Fried for others, or for any help or questions about the code.

| Number | Name | Command |
| ------ | ---- | ------- |
| S6 | Supervised: SMM, generative |  `./run_crosstask_i3d-resnet-audio.sh pca_semimarkov_sup --classifier semimarkov --training supervised` |
| U7 | HSMM + Narr + Ord | `./run_crosstask_i3d-resnet-audio.sh pca_semimarkov_unsup_narration_ordering --classifier semimarkov --training unsupervised --mix_tasks --task_specific_steps --sm_constrain_transitions --annotate_background_with_previous --sm_constrain_with_narration train --sm_constrain_narration_weight=-1e4 --cuda` |

## Credits

- Parts of the data loading and evaluation code are based on [this repo](https://github.com/Annusha/slim_mallow) from Anna Kukleva.
- Code for invertible emission distributions are based on Junxian He's [structured flow code](https://github.com/jxhe/struct-learning-with-flow). (These didn't make it into the paper -- I wasn't able to get them to work consistently better than Gaussian emissions over the PCA features.)
- Compound HSMM / VAE models are based on Yoon Kim's [Compound PCFG code](https://github.com/harvardnlp/compound-pcfg). (These also didn't make it into the paper, for the same reasons.)
