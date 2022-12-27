# Feature Attribution Explanation to Detect Harmful Dataset Shift

Detecting whether a distribution shift has occurred in the data is an important but easily overlooked step when testing, and seemingly small changes in the data distribution may largely affect the performance of the classifier. In this work, we focus on detecting harmful dataset shifts, i.e., shifts that are detrimental to the performance of the classifier. Based on the dataset shift detection framework proposed by Rabanser et al., we use feature attribution explanation (FAE) methods as “dimensionality reduction techniques" to use the gradient information in the model, and a multivariate two-sample detection technique called maximum mean discrepancy (MMD) to detect dataset shifts. The results of experiments using more than twenty shifts on three widely used image datasets show that the feature attribution explanation methods are more effective in identifying harmful data shifts than existing methods. Moreover, experiments on several models of different types/structures show that the ability of our method to identify harmful shifts is virtually indistinguishable across models, i.e., its detection ability is independent of the model used.

Code is based on "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift": (https://github.com/steverab/failing-loudly) and "Detecting Covariate Drift with Explanations" (https://github.com/DFKI-NLP/xai-shift-detection)

## Running experiments

Run experiments using:

```
python pipeline.py Dataset Shift_Type multiv Model_Name
```

Example: `python pipeline.py mnist adversarial_shift multiv resnet50`



### Dependencies

We require the following dependencies:
- `keras`: https://github.com/keras-team/keras
- `tensorflow`: https://github.com/tensorflow/tensorflow
- `pytorch`: https://github.com/pytorch/pytorch
- `sklearn`: https://github.com/scikit-learn/scikit-learn
- `matplotlib`: https://github.com/matplotlib/matplotlib
- `torch-two-sample`: https://github.com/josipd/torch-two-sample
- `keras-resnet`: https://github.com/broadinstitute/keras-resnet
- `alibi-detect`: https://github.com/SeldonIO/alibi-detect
