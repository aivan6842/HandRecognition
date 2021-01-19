# HandRecognition
This is a project in which I used PyTorch to classify images of a hand and label them with the number of fingers being held up. The goal of this project was to explore the PyTorch
module by creating, training and utilizing a model. I was able to so by using transfer learning on the ResNet-18 model and only modifying the last layer. 

# Transfer Learning
To start of this project I knew that I would not have a lot of data and building a convolutional neural network might not be the best idea since it would require a decent amount
of data. This is where transfer learning helped. I ended up using ResNet-18 as a feature extractor and only modifying the last layer to match the number of classes that I had(5).
This helped in a number of ways. Firstly, the training time was much shorter than it would have been to train a full ResNet-18 model. Since I was only modifying the last layer, I
didn't need to compute gradients or the optimal weights and baises for all the parameters of the previous layers. This computation would only have to be done for the last layer.
As mentioned, perhaps the best feature of transfer learning was that it didn't need as much data as usual. I was able to train an ok model with roughly 150 training images for
each class.

# Results
The network did fairly well in classifying all numbers except the number 2. I'm not 100% sure where the problem stems from but I did try to apply numerous different transforms to
the data as well as change some of the hyperparameters, but nothing seemed to be fixing this issue. My last speculation is that it might have something to do with the data that
the model was trained on. Overall, with the limited data the model achieved a 79% accuracy for this classification problem. The result can be summarized by the following confusion
matrix
![Confusion Matrix](https://github.com/aivan6842/HandRecognition/blob/master/Images/confusion_matrix.png)

# The Included Files
There are 4 main files included:
  * ```Hand Tracking.ipynb```: This is a script which you can interactively classify real time images of your own hand! Once the script is run, the green square in the middle
                             is where you can place your hand. Pressing ```y``` will take a snapshot and classify the image.
  * ```modelParams.pth```: This is file which contains the parameters for the model which I have trained. You can load them into any script that you want. There should be an example
                        of this in ```Hand Tracking.ipynb```.
  * ```TrainingDataCollector.ipynb```: This is a simple file which saves training data to a directory on your computer. This is what I used to gather training data and you can use it too!
  * ```TransferLearn.ipynb```: This is the file where the transfer learning and training takes place. This is also where I save and evaluate the model.
