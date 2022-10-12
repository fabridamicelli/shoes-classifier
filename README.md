# Shoes classifier
This is a little beginner computer vision project to learn a bit about it using the [fastai library](https://docs.fast.ai/), [timm pytorch models](https://github.com/rwightman/pytorch-image-models) and the [nike-adidas-and-converse-imaged kaggle dataset](https://www.kaggle.com/datasets/die9origephit/nike-adidas-and-converse-imaged) as well as scraped images.

## TOC
- [Experiments](01_training.ipynb): Run a few quick experiments to achieve a sort of "good enough" results for a first iteration
- [Evaluate model](02_prediction_valid.ipynb): Evaluate model on whole validation split
- [Extend dataset](03_extend_dataset.ipynb): Create a new test dataset by downloading (a few hundred) photos from the internet
- [Evaluate model](04_prediction_test.ipynb): Evaluate model on the test dataset

## Run the code
The easiest way is to create a conda virtual environment:
```shell
conda create -f environment.yml
```

The names of the notebooks reflects the order of the work I did.

## Main Thoughts/Ideas
In this notebooks I document a handful of quick experiments that I ran trying to solve this task.
As you will see, there is nothing crazy systematic going on, just a few tries applying the heuristic of starting with a model that performs acceptably in general and can be trained fast.

After inspecting the dataset I decided to experiment with a couple of augmentations (rotation, zoom).
I tried two sizes of a `convnext` architecture, the smaller one for quick experiments, the larger for final scaling. 

Overall my impression is that this task is relatively simple and requires no so much fiddling around and/or sophisticated techniques.
For example the augmentations showed relatively unstable performance improvements and having a larger network pretty much equates the best performance achieved with the smaller net. 
Also, as it can be seen in the experiments, the fine-tuning was almost already enough to get the maximum performance that I achieved in the experiments.
I don't know exactly on which original dataset the `convnext` networks were trained, but I can imagine there were similarities with the dataset at hand here.

By looking at the confusion matrix and the test-time-augmented error rate I judged the performance was OK so I just decided to keep things simple and to stick to the larger net for inference on the test split.

A few examples were still wrongly classified and, interestingly enough, almost all models I tried got at least some of those example wrong (so that might be an interesting aspect to look into in the future).
I also ran longer trainings (more epochs), but no meaningful improvement were seen.

Since I ran all the experiments with this train/valid split provided by the kaggle dataset, I then wanted to test how well the model generalizes. 
For that, I collected (web scraping) new photos that I used as a test split after finishing the experiments.
I was surprised with the model's ability to generalize to unseen data, as we could see on the test-split results.


## A few other takeaways/learnings**
Rotation augmentation: Rotating the images seemed to be an obviously benefitial augmentation to me a priori because many images are themselves rotated in the original dataset.
Still, I found it was pretty easy to mess up things by rotating too much (eg 60°).

Learning rate tuning: It is well known that this hyperparameter is absolutely critical. 
I thought I could make a better job of fine-tuning by doing some one-cycle frozen epochs and then re-running the fastai's `LRFinder` to adjust it more carefully after unfreezing and setting some tailored differential learning rates.
Some experiments had better results, but not very consistently. Take away: the built-in heuristic to adjust the learning rate in the fastai's `fine_tune` method is pretty good (at least for this dataset and these architectures)!

## Further work
- [] analize similarity (are we cheating by looking at same image twice (after download)?)
- [] randomized crop augmentation might be worth trying
- [] The model seems to still make some mistakes on what to my eyes are "easy" examples. It may be worth investigating why.


## Points to look up and further read
- What are `pct_start` and `div` in `learn.fit_one_cycle`? Which role do they play?
- Look at groups of layers, investigate how the different lrs affect them
