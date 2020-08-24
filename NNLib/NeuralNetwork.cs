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

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("NNLibXUnitTest")]

namespace NNLib
{
    public class NeuralNetwork : IXmlSerializable
    {
        private readonly List<Layer> layers = new List<Layer>();
        private ILossLayer loss = null;
        private IOptimizer optimizer = null;

        private bool compiled = false;
        private int? sizeOfMiniBatch = null;

        public Tensor LastPrediction { get; private set; }

        private void DimCheck(ref int? inDim, int? expected)
        {
            if (inDim != null)
            {
                if (inDim != expected)
                    throw new FormatException("Added layer doesn't fit expected dimensions!");
            }
            else
            {
                inDim = expected;
            }
        }

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
                if (layers[^1] is IWithActivation withActivation && withActivation.ActivationUsed == ActivationFunctions.Softmax)
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

        public void Compile(ILossLayer loss, IOptimizer optimizer)
        {
            this.loss = loss ?? throw new ArgumentException("Loss must be specified!");
            this.optimizer = optimizer ?? throw new ArgumentException("Optimizer must be specified!");

            var withActivation = layers[^1] as IWithActivation;
            if (loss is SparseCategoricalCrossEntropy && (withActivation == null || withActivation.ActivationUsed != ActivationFunctions.Softmax))
                throw new InvalidOperationException("SparseCategoricalCrossEntropy loss is supported only when preceded by Softmax!");

            if (withActivation != null && withActivation.ActivationUsed == ActivationFunctions.Softmax && !(loss is SparseCategoricalCrossEntropy))
                throw new InvalidOperationException("Last layer with activation Softmax is supported only when followed by SparseCategoricalCrossEntropy!");

            foreach (var layer in layers)
            {
                layer.Compile();
                optimizer.AddLayer(layer);
            }

            compiled = true;
        }

        public Tensor Predict(Tensor input)
        {
            if (input.Rows != layers[0].InRows || input.Columns != layers[0].InColumns || input.Depth != layers[0].InDepth)
                throw new ArgumentException("Input dimensions doesn't match first layers inDimensions !");

            Tensor currentOutput = input;
            foreach (var layer in layers)
                currentOutput = layer.ForwardPass(currentOutput);

            return currentOutput;
        }

        public double Forward(Tensor input, Tensor expectedOutput)
        {
            if (input.Rows != layers[0].InRows)
                throw new ArgumentException("Input dimensions doesn't match first layers inDimensions !");

            Tensor currentOutput = input;
            foreach (var layer in layers)
                currentOutput = layer.ForwardPass(currentOutput);

            LastPrediction = currentOutput;
            return loss.ForwardPass(currentOutput, expectedOutput);
        }

        public void Backward()
        {
            Tensor currentGradient = loss.BackwardPass();

            for (int i = layers.Count - 1; i >= 0; i--)
            {
                currentGradient = layers[i].BackwardPass(currentGradient, out Tensor gradientWeights, out Tensor gradientBias);
                optimizer.RememberGradient(i, gradientWeights, gradientBias);
            }

            sizeOfMiniBatch ??= 0;
            sizeOfMiniBatch++;
        }

        public void Fit(IDataset dataset, int epochs, int batchSize)
        {
            var epochNumber = 1;
            dataset.Epochs = epochs;
            dataset.BatchSize = batchSize;
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

                    Console.WriteLine($"Epoch {epochNumber++} loss : {loss} accuracy : {accuracy}");
                }
            }
        }

        public void Evaluate(IDataset dataset)
        {
            var (testInputs, testLabels) = dataset.GetTestSet();
            Console.WriteLine("Started evaluation...");
            var loss = Forward(testInputs, testLabels);

            var accuracy = GetAccuracy(testLabels, LastPrediction);

            Console.WriteLine($"Loss : {loss} accuracy : {accuracy}");
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

        public void UpdateWeights()
        {
            optimizer.CalculateAndUpdateWeights((int)sizeOfMiniBatch, layers);
            sizeOfMiniBatch = null;
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
        }

        public void WriteXml(XmlWriter writer)
        {
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
