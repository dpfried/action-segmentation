# Learning to Segment Actions from Observation and Narration  

Code for the paper:  
[Learning to Segment Actions from Observation and Narration](https://arxiv.org/abs/2005.03684)  
Daniel Fried, Jean-Baptiste Alayrac, Phil Blunsom, Chris Dyer, Stephen Clark, and Aida Nematzadeh  
ACL, 2020

## Summary

This repository provides a system for segmenting and labeling actions in a video, using a simple generative segmental (hidden semi-Markov) model of the video. This model can be used as a strong baseline for action segmentation on instructional video datasets such as [CrossTask](https://github.com/DmZhukov/CrossTask) ([Zhukov et al., CVPR 2019](https://arxiv.org/abs/1903.08225)), and can be trained fully supervised (with action labels for each frame in each video) or with weak supervision from narrative descriptions and "canonical" step orderings. Please see our paper for more details.

## Requirements

* pytorch 1.2
* Our fork of [pytorch-struct](https://github.com/dpfried/pytorch-struct-hsmm). (Newer versions may run out of memory on the long videos in the CrossTask dataset, due to changes to pytorch-struct that improve runtime complexity but increase memory usage.)

See `env.yml` for a full list of dependencies, which can be installed with conda.

## Documentation

Instructions / documentation coming (hopefully!) soon. If you're interested in running the code in the meantime, please contact Daniel Fried.

## Credits

- Parts of the data loading and evaluation code are based on [this repo](https://github.com/Annusha/slim_mallow) from Anna Kukleva.
- Code for invertible emission distributions are based on Junxian He's [structured flow code](https://github.com/jxhe/struct-learning-with-flow). (These didn't make it into the paper -- I wasn't able to get them to work consistently better than Gaussian emissions over the PCA features.)
- Compound HSMM / VAE models are based on Yoon Kim's [Compound PCFG code](https://github.com/harvardnlp/compound-pcfg). (These also didn't make it into the paper, for the same reasons.)
