# i4c-2026

You will receive a training dataset of paired images: for each sample, you get a degraded image (noisy + low resolution) and the corresponding ground truth image (clean + full resolution). Your job is to train an AI model that learns to reverse the degradation taking a bad image as input and producing a restored image that matches the ground truth as closely as possible.

    Your model must handle ALL degradation types simultaneously a single image may have speckle noise AND reduced resolution at the same time.
    The test set will include images from different sources than the training data (out-of-distribution). Your model must generalize not just memorize the training examples.
    Speed matters. Your model will be benchmarked on inference time. A model that produces great results but takes 10 minutes per image is less useful than one that produces good results in 10 seconds.

    Important data notes:

    The degraded image intensity range may EXCEED the ground truth range this is expected behaviour caused by speckle noise pushing pixel values beyond the original signal. Your model must handle this.
    The images come from diverse data origins different types of semiconductor structures. Your model should generalize across these variations, not overfit to one type.
    Images are grayscale (single channel). Colour images are NOT part of this challenge.


Test Data — What Comes Later

After the training phase, KLA will release a test dataset. The test set contains:

    In-distribution samples: images similar to what you trained on. Tests accuracy.
    Out-of-distribution samples: images from different sources than the training data. Tests generalization and robustness  whether your model can handle image types it has never seen
