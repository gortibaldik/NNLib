using System;
using System.Globalization;
using System.Text;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;
using NNLib.Activations;
using NNLib;

namespace NNLib.Layers
{
    public class DenseLayer : Layer, ITrainable, IWithActivation, IXmlSerializable
    {
        public override int OutDepth { get => 1; }
        public override int OutColumns { get => InColumns.Value; }

        private Tensor _weights = null;
        private Tensor _bias = null;

        private IActivationLayer _activation = null;
        private NInitializer _weightInit = null;
        private NInitializer _biasInit = null;

        private Tensor lastInput = null;
        private Tensor lastOutput = null;

        // only for serialization - deserialization purposes
        private DenseLayer() { }

        /// <summary>
        /// Creates new dense layer.
        /// </summary>
        /// <param name="outRows">Number of neurons in the DenseLayer</param>
        /// <param name="activation">If not specified, LinearActivation is used</param>
        /// <param name="weightInit">If not specified, NInitGlorotUniform is used</param>
        /// <param name="biasInit">If not specified, NInitZero is used</param>
        public DenseLayer(int outRows, IActivationLayer activation = null, NInitializer weightInit = null, NInitializer biasInit = null)
        {
            _activation = activation ?? new LinearActivation();
            _weightInit = weightInit;
            _biasInit = biasInit;
            OutRows = outRows > 0 ? outRows : throw new ArgumentException("Weight dimensions must be greater than 0!");

            // InDimensions kept null until NeuralNetwork class uses internal setter
        }

        /// <summary>
        /// Creates new dense layer.
        /// </summary>
        /// <param name="inputShape">Shape of the input vector</param>
        /// <param name="outRows">Number of neurons in the DenseLayer</param>
        /// <param name="activation">If not specified, LinearActivation is used</param>
        /// <param name="weightInit">If not specified, NInitGlorotUniform is used</param>
        /// <param name="biasInit">If not specified, NInitZero is used</param>
        public DenseLayer(Shape inputShape, int outRows, IActivationLayer activation = null, NInitializer weightInit = null, NInitializer biasInit = null) : this(outRows, activation, weightInit, biasInit)
        {
            InRows = inputShape.Rows;
            InColumns = inputShape.Columns;
            InDepth = inputShape.Depth;
        }

        /// <summary>
        /// Creates new DenseLayer with specified weights and bias. The tensors are used by reference. 
        /// </summary>
        /// <param name="weights">Cannot be null</param>
        /// <param name="bias">Can be null</param>
        /// <param name="activation">If null Linear activation is used</param>
        public DenseLayer(Tensor weights, Tensor bias, IActivationLayer activation)
        {
            if (weights == null)
                throw new ArgumentException("Cannot create dense layer without weights !");

            if (bias == null)
                _bias = new Tensor(weights.BatchSize, weights.Depth, weights.Rows, 1);
            else if (bias.Rows == weights.Rows)
                _bias = bias;
            else
                throw new ArgumentException($"{nameof(bias)}.{nameof(bias.Rows)} : {bias.Rows} doesn't correspond to {nameof(weights)}.{nameof(weights.Columns)} : {weights.Columns}");

            _weights = weights;
            InRows = _weights.Columns;
            OutRows = _weights.Rows;
            _activation = activation ?? new LinearActivation();
        }

        public override void Compile()
        {
            if (InRows == 0)
                throw new InvalidOperationException("Input dimension hasn't been declared yet!");

            if (_weights == null)
            {
                if (_weightInit == null)
                    _weightInit = NeuronInitializers.NInitGlorotUniform(OutRows, (int)InRows);

                if (_biasInit == null)
                    _biasInit = NeuronInitializers.NInitZero;

                // W*I + B = O
                // where W - _weights, B - _bias, I - input, O - output
                // therefore outputDimension
                _weights = new Tensor(1, 1, OutRows, (int)InRows, _weightInit)
                {
                    Mode = TensorMultiplicationModes.LastLevel
                };

                // bias isn't used for multiplication only for column-wise
                // addition, therefore names OutDim, InDim don't have any special
                // meaning for bias
                _bias = new Tensor(1, 1, OutRows, 1, _biasInit)
                {
                    Mode = TensorMultiplicationModes.LastLevel
                };
            }

            compiled = true;
        }

        public override Tensor ForwardPass(Tensor resultPrevious, bool training = false)
        {
            InputCheck(resultPrevious);
            forwardPerformed = true;

            lastInput = resultPrevious;
            lastOutput = _weights * resultPrevious + _bias;
            Tensor result = (_activation == null) ? lastOutput : _activation.ForwardPass(lastOutput);
            return result;
        }

        public override Tensor BackwardPass(Tensor previousGradient, out Tensor derivativeWeights, out Tensor derivativeBias)
        {
            InputCheck(input: previousGradient, fwd: false);
            if (lastInput == null || !forwardPerformed)
                throw new InvalidOperationException("No forward pass before backward pass !");

            previousGradient = _activation == null ? previousGradient : _activation.BackwardPass(previousGradient);

            // out parameters .BatchSize should be 1 so only the average across all the batches is returned
            derivativeWeights = (1D/previousGradient.BatchSize) * (previousGradient * lastInput.Transpose()).SumBatch();
            derivativeBias = _bias == null ? null : (1D / previousGradient.BatchSize) * previousGradient.SumBatch().SumRows();

            // gradient.BatchSize should equal previousGradient.BatchSize
            Tensor gradient = _weights.Transpose() * previousGradient;

            lastInput = null;
            forwardPerformed = false;
            return gradient;
        }

        void ITrainable.SetWeights(Tensor weights)
            => _weights = weights;

        void ITrainable.SetBias(Tensor bias)
            => _bias = bias;

        Tensor ITrainable.GetWeights()
            => _weights;

        Tensor ITrainable.GetBias()
            => _bias;

        string IWithActivation.ActivationUsed { get => _activation.GetType().Name; }

        XmlSchema IXmlSerializable.GetSchema()
            => null;

        void IXmlSerializable.ReadXml(XmlReader reader)
        {
            // reads and sets input vector dimensions and output vector dimensions
            ReadXml(reader);
            var activationStr = reader.GetAttribute(nameof(_activation));
            _activation = (new ActivationFactory()).Create(activationStr);

            var weightsStr = reader.GetAttribute(nameof(_weights));
            var biasStr = reader.GetAttribute(nameof(_bias));

            if (weightsStr == null)
                throw new FormatException("Wrong xml representation, weights not present in the serialized layer! ");

            var data = new double[1][];
            data[0] = weightsStr.DeserializeIntoDoubleArray();
            _weights = new Tensor(1, OutRows, (int)InRows, data);

            if (biasStr != null)
            {
                data[0] = biasStr.DeserializeIntoDoubleArray();
                _bias = new Tensor(1, OutRows, 1, data);
            }

            reader.ReadStartElement();
        }

        void IXmlSerializable.WriteXml(XmlWriter writer)
        {
            // writes input and output dimensions
            WriteXml(writer);
            writer.WriteAttributeString(nameof(_activation), _activation.GetType().Name);

            if (_weights == null)
                throw new InvalidOperationException("Cannot serialize non compiled network !");

            writer.WriteAttributeString(nameof(_weights), _weights.SerializeToString());

            if (_bias != null)
                writer.WriteAttributeString(nameof(_bias), _bias.SerializeToString());
        }

        public override string ToString()
        {
            if (_weights == null)
                return "";

            if (_bias == null)
                return _weights.ToString();

            return _weights.ToString() + "\n and bias \n" + _bias.ToString();
        }

    }
}
