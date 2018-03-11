# densenet
Implements dense net on MNIST dataset

## Model

    A brief description of the Model is provided below.
 
### high level view
    
Input layer -> Block 1 -> Batch Normalization -> Relu -> Global average pooling -> Fully connected layer

* **Block 1** consists of 4 alternating combinations of dense layer and transition layer

* **Dense layer** is made of the following sequence:
   1. Batch Normalization
   2. Relu
   3. Convolutional 2d layer
   4. Concatination of the previous layers output to the previous element(Convolutional 2d layer)
 
* **Transition Layer** is made of the following sequence:
   1. Batch Normalization 
   2. Relu
   3. Convolutional 2d layer
   4. Average Pooling 
      
### hyper parameters and other essential attributes

* Input dimension = [100, 28, 28, 1] (Trained using batches of 100 images)
* Ouput dimension = [10]
* epoch = 3000      
