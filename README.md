# Sequential neural network

The aim of this project is to implement functional environment for creating sequential neural networks with ease.

## Usage

### 1. How to create and use the network
____________________________
The main part of the environment is the class ```NeuralNetwork``` which holds all the layers and is responsible for training (fitting created network on a dataset) and evaluating the performance of the neural network on test data.
#### Adding layers to neural network:
There are currently 3 types of available layers : ```DenseLayer```, ```FlattenLayer``` and ```InputLayer```.
The first layer must have specified input dimensions. If any other layers input shape is specified it must match the last networks layers output shape. The network checks for equality of references, therefore it isn't possible to add one instance of any layer multiple times.

```csharp
var net = new NeuralNetwork();

// first layer -> input shape must be specified
net.Add(new Flatten(new Shape(1, 28, 28));

// other layers, only number of neurons is specified, we don't have to 
// specify the input shape
net.Add(new DenseLayer(100, new ReLU()));
net.Add(new DenseLayer(100, new ReLU()));
net.Add(new DenseLayer(10, new Softmax()));
```

#### Training and predicting:
There are 4 methods of neural network which the user can use :
```net.Compile(ILossLayer loss, IOptimizer optimizer)``` - prepares the network for training, using specified optimizer
and specified loss ___Warning : Network must have been compiled in order to perform Fit() and Evaluate()___

```net.Fit(IDataset dataset, int epochs, int batchSize)``` - fits the model on the training data from the specified ```IDataset``` ([more on IDataset](#interface-IDataset)) and returns the statistics of the model on the validation part of ```IDataset```

```net.Evaluate(IDataset dataset)``` - returns the statistics (loss, accuracy) of the model on the test part of ```IDataset```

```net.Predict(Tensor tensor)``` - returns the output of the last layer of the neural network after inputting ```tensor``` into the first layer 
_In order to be able to use this method, the weights and activations of all the layers must have been initialized, therefore the user can use this method only after ```compilation``` or ```deserialization```_

#### Serializing neural network: 
After training/fitting the network on an ```IDataset``` there is a possibility to save the network. It implements ```IXmlSerializable``` so the ```weights``` of trainable layers as well as the input and output dimensions and the structure of the network can be saved to ```XML``` for later use.
```csharp
// serializing
using var writer = new StreamWriter("savedNetwork.net");
var serializer = new XmlSerializer(typeof(NeuralNetwork));
serializer.Serialize(writer, net);

// deserializing
var serializer = new XmlSerializer(typeof(NeuralNetwork));
using (var reader = new StreamReader("savedNetwork.net"))
net = (NeuralNetwork)serializer.Deserialize(reader);
```

### 2. ```class Tensor```

The most important class of the library. All the ```weights``` and ```biases``` as well as all the training data and labels, the neural network inputs and outputs are stored as ```Tensors```. The class aims to provide fast implementation of the most important (from the perspective of the neural network) operations on the ```Tensors``` :
- element-wise operations such as : Addition, Subtraction
- matrix-wise operation such as : Multiplication
- singleTensor operations such as : applying functor on all the elements of the tensor
- reshape and transpose operations

The computations on the tensor are performed using ```Task.Parallel``` library and the internal structure of the tensor is one ```double[]```. The ```Tensor```aims to be externally immutable. The indexer exposes only getter and all the operations return new ```Ten 

##### Modes of addition and subtraction :
- every tensor is 4 dimensional, therefore many questions emerge: 
	- How to perform tensor-tensor addition/subtraction if the tensors have different number of rows ? :
	this isn't supported and an ```InvalidOperationException``` is thrown if it happens
    
    - How to perform tensor-tensor addition/subtraction if the tensors have different number of columns ? :
    Since this is scenario in ```DenseLayer``` where ```bias.Columns == 1``` and ```bias``` is added to the 
    ```weights*input``` the situation is supported but only in the case when the right operand has 1 column. 
    Any other scenario is not supported and ```InvalidOperationException``` is thrown
    
    - How to perform tensor-tensor addition/subtraction if the tensors have different number of depth layers ?
    this isn't supported and an ```InvalidOperationException``` is thrown if it happens
    
    - How to perform tensor-tensor addition/subtraction if the tensors have different number of BatchSize layers ?
    Another very common scenario in the neural network, when the layer processes the batch of images although it
    has only 1 layer of ```weights```.  Supported only if the right operand has 1 layer of ```BatchSize``` and then
    the layer is added/subtracted from all the layers of the left operand.
    Any other scenario is not supported and ```InvalidOperationException``` is thrown
    
##### Modes of multiplication :
- the multiplication is performed only on the matrix part of the tensor

- the basic rules of matrix multiplication are enforced : if the operands are ```t1 * t2``` then ```t1.Columns == t2.Rows``` otherwise ```InvalidOperationException``` is thrown

- again, in case of ```DenseLayer```, ```input``` is multiplied from right side by ```weights``` where ```weights.Depth == weights.BatchSize == 1``` but ```input.Depth``` and ```input.BatchSize``` are any positive integers. 
Therefore there are 4 allowed modes of ```tensor``` multiplication : if the operands are ```t1 * t2``` then allowed combinations of dimensions include :
	- ```t1.Depth == t2.Depth && t1.BatchSize == t2.BatchSize```
	- ```t1.Depth == t2.Depth && t1.BatchSize == 1```
	- ```t1.Depth == 1 && t1.BatchSize == t2.BatchSize```
	- ```t1.Depth == 1 && t1.BatchSize == 1```
	- if ```t1.Depth == 1``` then the only matrix of the ```BatchSize``` layer of is used as left side operand for all the multiplications with all the ```depth``` layers of ```t2````
	- the same applies for ```t1.BatchSize == 1```

### 3. ```interface IDataset```

Any class aiming to provide data for network training must implement ```interface IDataset```. There are 4 properties
and 3 methods needed to be implemented :
- properties :
	- ```int Epochs {get; set;}```
		- set up by the neural network at the start of ```network.Fit()```. The number of traversals through the training part of the dataset during network training.
	- ```int BatchSize {ge; set;}```
	 	- set by the neural network at the start of ```network.Fit()```. The batchSize of the datapoints provided from the training part of the dataset
	- ```bool EndEpoch {get;}```
		- the flag to signalize the end of epoch, used during ```network.Fit()```
	- ```bool EndTraining {get`}```
		- the flag to signalize the end of training, used duringn ```network.Fit()```
- methods :
	- ```(Tensor Input, Tensor Label) GetBatch();```
		- the dataset should provide 2 tensors with ```tensor.BatchSize``` of ```IDataset.BatchSize``` with ```BatchSize``` number of dataPoints in ```Tensor input``` and ```BatchSize``` number of dataLabels in ```Tensor Label```
	- ```(Tensor Input, Tensor Label) GetValidation();```
		- the dataset should provide 2 tensors with all the validation dataPoints stacked in ```BatchSize``` dimension of ```Tensor Input``` and all validation dataLabels in ```BatchSize``` dimension of ```Tensor Label```
	- ```(Tensor Input, Tensor Label) GetTestSet();```
 		- the dataset should provide 2 tensors with all the test dataPoints stacked in ```BatchSize``` dimension of ```Tensor Input``` and all test dataLabels in ```BatchSize``` dimension of ```Tensor Label```
