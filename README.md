# Predicting landmarks using Convolutional Neural Networks (CNN)
This project (nd101-c2-landmarks-starter) is about using Convolutional Neural Network (CNN) module to build a landmark classifier. The model should automatically be able to predict the location of the image based on any landmarks depicted in the image. This project is the second requirement to graduate from Udacity's Deep learning Nano-degree.

The Project Steps:
The high level steps of the project include:
1.	Create a CNN to Classify Landmarks (from Scratch) - Here, you'll visualize the dataset, process it for training, and then build a convolutional neural network from scratch to classify the landmarks. You'll also describe some of your decisions around data processing and how you chose your network architecture.
2.	Create a CNN to Classify Landmarks (using Transfer Learning) - Next, you'll investigate different pre-trained models and decide on one to use for this classification task. Along with training and testing this transfer-learned network, you'll explain how you arrived at the pre-trained network you chose.
3.	Write Your Landmark Prediction Algorithm - Finally, you will use your best model to create a simple interface for others to be able to use your model to find the most likely landmarks depicted in an image. You'll also test out your model yourself and reflect on the strengths and weaknesses of your model.
The detailed project steps are included within the project notebook linked further below.

My code passed all the unit tests as well as the rubric for all the three parts.
Part 1:
Net(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=50176, out_features=500, bias=True)
  (fc2): Linear(in_features=500, out_features=50, bias=True)
  (dropout): Dropout(p=0.25)
)
Hyperparameters: SGD optimizer, lr=0.01, CrossEntropyLoss
Parameters: batch_size = 20, epochs = 100 
My output:
Training Loss: 0.000415 	Validation Loss: 0.000537
Test Loss: 2.299584		Test Accuracy: 44% (554/1250)

Part 2:
Net(Pretrained Vgg16 model)
Hyperparameters: SGD optimizer, lr=0.001, CrossEntropyLoss
Parameters: batch_size = 20, epochs = 100 
My output:
Training Loss: 0.000160 	Validation Loss: 0.000226
Test Loss: 0.779844		Test Accuracy: 78% (984/1250)
