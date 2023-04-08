# Feature Attribution Explanation to Detect Harmful Dataset Shift

This is the code for the paper "Feature Attribution Explanation to Detect Harmful Dataset Shift" in International Joint Conference on Neural Networks (IJCNN) 2023, in which we proposed a method that combines feature attribution explanation and two sample tests to detect harmful dataset shifts.

### Personal Use Only. No Commercial Use.

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
