using System;
using NNLib.Activations;
using NNLib.Optimizers;

namespace NNLib.Layers
{
    public class DenseLayer : Layer, ITrainable, IWithActivation
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

        public DenseLayer(int outRows, IActivationLayer activation = null, NInitializer weightInit = null, NInitializer biasInit = null)
        {
            _activation = activation == null ? new LinearActivation() : activation;
            _weightInit = weightInit == null ? NeuronInitializers.NInitNormal : weightInit;
            _biasInit = biasInit == null ? NeuronInitializers.NInitZero : biasInit;
            OutRows = outRows > 0 ? outRows : throw new ArgumentException("Weight dimensions must be greater than 0!");

            // InDimensions kept null until NeuralNetwork class uses internal setter
        }

        public DenseLayer(int inRows, int outDim, NInitializer weightInit, NInitializer biasInit, IActivationLayer activation) : this(outDim, activation, weightInit, biasInit)
        {
            InRows = inRows > 0 ? inRows : throw new ArgumentException("Weight dimensions must be greater than 0!");
        }

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
            _activation = activation == null ? new LinearActivation() : activation;
        }

        public override void Compile()
        {
            if (InRows == 0)
                throw new InvalidOperationException("Input dimension hasn't been declared yet!");

            if (_weights == null)
            {
                // W*I + B = O
                // where W - _weights, B - _bias, I - input, O - output
                // therefore outputDimension
                _weights = new Tensor(1, 1, OutRows, (int)InRows, _weightInit);

                // bias isn't used for multiplication only for column-wise
                // addition, therefore names OutDim, InDim don't have any special
                // meaning for bias
                _bias = new Tensor(1, 1, OutRows, 1, _biasInit);
            }

            compiled = true;
        }

        public override Tensor ForwardPass(Tensor resultPrevious, bool training = false)
        {
            InputCheck(resultPrevious);
            lastInput = resultPrevious;
            lastOutput = _weights * resultPrevious + _bias;
            Tensor result = (_activation == null) ? lastOutput : _activation.ForwardPass(lastOutput);
            return result;
        }

        public override Tensor BackwardPass(Tensor previousGradient, out Tensor derivativeWeights, out Tensor derivativeBias)
        {
            InputCheck(input: previousGradient, fwd: false);
            if (lastInput == null)
                throw new InvalidOperationException("No forward pass before backward pass !");

            previousGradient = _activation == null ? previousGradient : _activation.BackwardPass(previousGradient);
            derivativeWeights = (1/previousGradient.BatchSize) * (previousGradient * lastInput.Transpose()).SumBatch();
            derivativeBias = _bias == null ? null : (1 / previousGradient.BatchSize) * previousGradient.SumBatch().SumRows();
            Tensor gradient = _weights.Transpose() * previousGradient;
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

        ActivationFunctions IWithActivation.ActivationUsed { get => _activation.Name; }
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
