using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;
using NNLib.Activations;
using NNLib.Layers;
using NNLib.Losses;
using NNLib.Optimizers;
using NNLib.Datasets;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("NNLibXUnitTest")]
[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("NNMnist")]

namespace NNLib
{
    /// <summary>
    /// Basic sequential NeuralNetwork.
    /// </summary>
    public class NeuralNetwork : IXmlSerializable
    {
        private readonly List<Layer> layers = new List<Layer>();
        private ILossLayer loss = null;
        private IOptimizer optimizer = null;

        private bool compiled = false;
        private bool deserialized = false;
        private bool forwardPerformed = false;
        private int? sizeOfMiniBatch = null;

        /// <summary>
        /// Provides the last output of the neural network
        /// </summary>
        public Tensor LastPrediction { get; private set; } = null;

        /// <summary>
        /// If actual is null, then its value is set to expected
        /// otherwise check for actual == expected is performed
        /// if it fails, FormatException is thrown
        /// </summary>
        private void DimCheck(ref int? actual, int expected)
        {
            if (actual != null)
            {
                if (actual != expected)
                    throw new FormatException("Added layer doesn't fit expected dimensions!");
            }
            else
            {
                actual = expected;
            }
        }

        /// <summary>
        /// Append the layer at the end of the neural network.
        ///
        /// In case of the first layer the input dimensions must be specified
        /// In case of all the other layers, if some of the input dimensions are
        /// specified they must match the current output dimensions of the network,
        /// otherwise InvalidOperationException is thrown.
        ///
        /// Since the network is sequential check if the added layer isn't already
        /// part of the network is performed.
        /// 
        /// There can be at most 1 layer with softmax activation in the neural network
        /// and it must be the last one. Compilation is then supported only with 
        /// SparseCategoricalCrossEntropy loss.
        /// </summary>
        public void Add(Layer layer)
        {
            if (compiled)
                compiled = false;

            if (layers.Count == 0)
            {
                if (layer.InDepth == null || layer.InRows == null || layer.InColumns == null)
                    throw new ArgumentException("You must specify the input shape for the first layer !");
            }
            else
            {
                for (int i = 0; i < layers.Count; i++)
                    if (object.ReferenceEquals(layers[i], layer) )
                        throw new InvalidOperationException("Cannot add one layer more times to the sequential network !");

                if (layers[^1] is IWithActivation withActivation && withActivation.ActivationUsed == nameof(Softmax))
                    throw new InvalidOperationException("Mode with adding more layers after softmax is not supported!");

                // checking initialized and initializing unitialized dimensions
                int? variable = layer.InDepth;
                DimCheck(ref variable, layers[^1].OutDepth);
                layer.InDepth = variable;

                variable = layer.InRows;
                DimCheck(ref variable, layers[^1].OutRows);
                layer.InRows = variable;

                variable = layer.InColumns;
                DimCheck(ref variable, layers[^1].OutColumns);
                layer.InColumns = variable;
            }

            layers.Add(layer);
        }

        /// <summary>
        /// Prepares the NeuralNetwork for Forward and Backward usage, methods Fit and Evaluate.
        /// Checks for the last layer, if it is softmax, then it must be followed by SparseCategoricalCrossEntropy
        /// loss function.
        /// </summary>
        public void Compile(ILossLayer loss, IOptimizer optimizer)
        {
            this.loss = loss ?? throw new ArgumentException("Loss must be specified!");
            this.optimizer = optimizer ?? throw new ArgumentException("Optimizer must be specified!");

            var withActivation = layers[^1] as IWithActivation;
            if (loss is SparseCategoricalCrossEntropy && (withActivation == null || withActivation.ActivationUsed != nameof(Softmax)))
                throw new InvalidOperationException("SparseCategoricalCrossEntropy loss is supported only when preceded by Softmax!");

            if (withActivation != null && withActivation.ActivationUsed == nameof(Softmax) && !(loss is SparseCategoricalCrossEntropy))
                throw new InvalidOperationException("Last layer with activation Softmax is supported only when followed by SparseCategoricalCrossEntropy!");

            foreach (var layer in layers)
            {
                layer.Compile();
                optimizer.AddLayer(layer);
            }

            optimizer.Compile();
            compiled = true;
        }

        /// <summary>
        /// Passes the input through all the layers and returns the result. (The input doesn't pass through loss)
        /// If we use the deserialized network, it doesn't have to be compiled. (Therefore we can predict from deserialized network without 
        /// optimizer or loss specified)
        /// </summary>
        public Tensor Predict(Tensor input)
        {
            if (!deserialized && !compiled)
                throw new InvalidOperationException("Cannot predict from the non-compiled, non-deserialized neural network !");

            if (input.Rows != layers[0].InRows || input.Columns != layers[0].InColumns || input.Depth != layers[0].InDepth)
                throw new ArgumentException("Input dimensions doesn't match first layers inDimensions !");

            Tensor currentOutput = input;
            foreach (var layer in layers)
                currentOutput = layer.ForwardPass(currentOutput);

            return currentOutput;
        }

        /// <summary>
        /// Passes the input through all the layers and the loss and returns the loss. 
        /// The network has to be compiled in order to perform the ForwardPass
        /// </summary>
        internal double Forward(Tensor input, Tensor expectedOutput)
        {
            if (!compiled)
                throw new InvalidOperationException("The neural network must be compiled in order to be able to perform the ForwardPass !");

            if (input.Rows != layers[0].InRows)
                throw new ArgumentException("Input dimensions doesn't match first layers inDimensions !");

            Tensor currentOutput = input;
            foreach (var layer in layers)
                currentOutput = layer.ForwardPass(currentOutput);

            LastPrediction = currentOutput;
            forwardPerformed = true;
            return loss.ForwardPass(currentOutput, expectedOutput);
        }

        /// <summary>
        /// Performs the backward pass through the neural network, remembers all the 
        /// gradients (the particular gradients are added together and the sizeOfMiniBatch
        /// keeps track of the number of Backward passes performed in order to update the
        /// weights correctly)
        /// 
        /// The forward pass must be performed before the backward pass and network must be 
        /// compiled in order to do it.
        /// </summary>
        internal void Backward()
        {
            if (!compiled)
                throw new InvalidOperationException("The neural network must be compiled in order to be able to perform the ForwardPass !");

            if (!forwardPerformed)
                throw new InvalidOperationException("Cannot perform BackwardPass before the forward pass !");
            
            forwardPerformed = false;

            Tensor currentGradient = loss.BackwardPass();

            for (int i = layers.Count - 1; i >= 0; i--)
            {
                currentGradient = layers[i].BackwardPass(currentGradient, out Tensor gradientWeights, out Tensor gradientBias);
                optimizer.RememberGradient(i, gradientWeights, gradientBias);
            }

            sizeOfMiniBatch ??= 0;
            sizeOfMiniBatch++;
        }

        /// <summary>
        /// Calls the optimizer specified during compilation to update all the weights of trainable layers.
        /// Throws InvalidOperationException if network hasn't been compiled yet or hasn't performed BackwardPass
        /// yet.
        /// </summary>
        internal void UpdateWeights()
        {
            if (sizeOfMiniBatch == null)
                throw new InvalidOperationException("Cannot update weights if the network hasn't been used yet !");

            optimizer.CalculateAndUpdateWeights((int)sizeOfMiniBatch, layers);
            sizeOfMiniBatch = null;
        }

        /// <summary>
        /// Models fits on the specified dataset. Training happens in batches
        /// of the specified batchSize and the dataset is traversed epochs times.
        /// </summary>
        public void Fit(IDataset dataset, int epochs, int batchSize)
        {
            var epochNumber = 1;
            dataset.Epochs = epochs > 0 ? epochs : throw new ArgumentOutOfRangeException("The number of epochs must be positive number !");
            dataset.BatchSize = batchSize > 0 ? batchSize : throw new ArgumentOutOfRangeException("The number of epochs must be positive number !");
            while (!dataset.EndTraining)
            {
                var (trainInputs, trainLabels) = dataset.GetBatch();
                if (trainInputs != null)
                {
                    Forward(trainInputs, trainLabels);
                    Backward();

                    UpdateWeights();
                }

                if (dataset.EndEpoch)
                {
                    var loss = 0D;
                    var (valInputs, valLabels) = dataset.GetValidation();
                    loss += Forward(valInputs, valLabels);

                    var accuracy = GetAccuracy(valLabels, LastPrediction);

                    Console.WriteLine($"Epoch {epochNumber++} validation loss : {loss} validation accuracy : {accuracy}");
                }
            }
        }

        /// <summary>
        /// Evaluates the performance (loss, accuracy) of the neural network on the test set of the dataset
        /// </summary>
        public void Evaluate(IDataset dataset)
        {
            var (testInputs, testLabels) = dataset.GetTestSet();
            Console.WriteLine("Started evaluation...");
            var loss = Forward(testInputs, testLabels);

            var accuracy = GetAccuracy(testLabels, LastPrediction);

            Console.WriteLine($"Test Loss : {loss} Test accuracy : {accuracy}");
        }

        private double GetAccuracy(Tensor trueDistro, Tensor probs)
        {
            var correct = 0;
            for (int b = 0; b < trueDistro.BatchSize; b++)
            {
                var max1 = -1D;
                var i1 = -1;
                var max2 = -1D;
                var i2 = -1;
                for (int r = 0; r < trueDistro.Rows; r++)
                {
                    if (trueDistro[b, 0, r, 0] > max1)
                    {
                        max1 = trueDistro[b, 0, r, 0];
                        i1 = r;
                    }
                    if (probs[b, 0, r, 0] > max2)
                    {
                        max2 = probs[b, 0, r, 0];
                        i2 = r;
                    }
                }
                if (i1 == i2)
                    correct++;
            }

            return (double)correct / trueDistro.BatchSize;
        }


        public override string ToString()
        {
            var builder = new StringBuilder();
            foreach (var layer in layers)
            {
                builder.Append(layer);
            }
            return builder.ToString();
        }

        public XmlSchema GetSchema()
            => null;

        public void ReadXml(XmlReader reader)
        {
            var layerFactory = new LayerFactory();
            reader.MoveToContent();
            
            // eats the attribute xml node and moves to the next one
            reader.ReadStartElement(nameof(NeuralNetwork));
            if (reader.NodeType == XmlNodeType.Whitespace)
                reader.MoveToContent();

            while(reader.NodeType != XmlNodeType.EndElement)
            {
                var layer = layerFactory.CreateLayer(reader.Name);
                var serializable = layer as IXmlSerializable;

                if (serializable != null)
                {
                    // obviously after serializable changes its properties
                    // on itself the changes reflect to layer
                    serializable.ReadXml(reader);
                    layers.Add(layer);
                }

                if (reader.NodeType == XmlNodeType.Whitespace)
                    reader.MoveToContent();
            }

            deserialized = true;
        }

        public void WriteXml(XmlWriter writer)
        {
            if (!compiled && !deserialized)
                throw new InvalidOperationException("Cannot serialize non-initialized network !");
                
            foreach (var layer in layers)
            {
                var serializable = layer as IXmlSerializable;
                if (serializable == null)
                {
                    var serializer = new XmlSerializer(layer.GetType());
                    serializer.Serialize(writer, layer);
                }
                else
                {
                    var type = layer.GetType().Name;
                    writer.WriteStartElement(type);
                    serializable.WriteXml(writer);
                    writer.WriteEndElement();
                }
            }
        }
    }
}
