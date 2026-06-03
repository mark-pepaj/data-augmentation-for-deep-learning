# Data Augmentation for Deep Learning

### Overview
Data augmentation is the process of applying various transformations to a data sample (i.g. an image) for the purpose of creating more data from already existing data.

***This can be useful when training Neural Networks.***

For example, when training a model to correctly classify images of handwritten digits, such as in the `MNIST` dataset, we should expect that the samples may very quite a lot. To allow the model to learn to recognize various differences in handwritten digits, we can apply certain transformations that teach the model to correctly generalize across different handwritings.

### Types of Transformations
There are many types of transformations that can be applied depending on the data being used for training.
<br>
To name a few:
  ```rotation, shear, color jitter, horizontal flip and vertical flip, random cropping, padding, scaling, etc.```

### The Augmentor Class
Here I've designed a class called `Augmentor` which makes augmenting data pretty simple.

#### Initialize an object of the Augmentor class
Various transformation may be passed at initialization:
```
augmentor = Augmentor(
        pad=2,
        augment_configs=[pad=2, RandomRotation(degrees=10), RandomAffine(scale=(0.9, 1.1), shear=5)]
        )
```



