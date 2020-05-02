Research code for the paper:

Learning to Segment Actions from Observation and Narration  
Daniel Fried, Jean-Baptiste Alayrac, Phil Blunsom, Chris Dyer, Stephen Clark, and Aida Nematzadeh  
ACL, 2020

Instructions / documentation coming (hopefully!) soon. If you're interested in running the code in the meantime, please contact Daniel Fried.

Credits:
- Parts of the data loading and evaluation code are based on [this repo](https://github.com/Annusha/slim_mallow) from Anna Kukleva.
- Code for invertible emission distributions are based on Junxian He's [structured flow code](https://github.com/jxhe/struct-learning-with-flow). (These didn't make it into the paper -- I wasn't able to get them to work consistently better than Gaussian emissions over the PCA features.)
- Compound HSMM / VAE models are based on Yoon Kim's [Compound PCFG code](https://github.com/harvardnlp/compound-pcfg). (These also didn't make it into the paper, for the same reasons.)
