using System;
using NNLib.Activations;

namespace NNLib.Layers
{
    public class DenseLayer : Layer, ITrainable
    {
        private Tensor _weights = null;
        private Tensor _bias = null;

        private IActivationLayer _activation = null;
        private NInitializer _weightInit = null;
        private NInitializer _biasInit = null;

        private Tensor lastInput = null;
        private Tensor lastOutput = null;

        public DenseLayer(int outDim, IActivationLayer activation = null, NInitializer weightInit = null, NInitializer biasInit = null)
        {
            _activation     = activation    ==  null ? new LinearActivation()            : activation;
            _weightInit     = weightInit    ==  null ? NeuronInitializers.NInitNormal    : weightInit;
            _biasInit       = biasInit      ==  null ? NeuronInitializers.NInitZero      : biasInit;
            OutDim          = outDim        >   0    ? outDim                            : throw new ArgumentException("Weight dimensions must be greater than 0!");

            // InDim is kept -1 until NeuralNetwork class uses internal setter
        }

        public DenseLayer(int inDim, int outDim, NInitializer weightInit, NInitializer biasInit, IActivationLayer activation) : this(outDim, activation, weightInit, biasInit)
        {
            InDim = inDim > 0 ? inDim : throw new ArgumentException("Weight dimensions must be greater than 0!");
        }

        public DenseLayer(Tensor weights, Tensor bias, IActivationLayer activation)
        {
            if (weights == null)
                throw new ArgumentException("Cannot create dense layer without weights !");

            if (bias == null)
                _bias = new Tensor(weights.Depth, weights.Rows, 1);
            else if (bias.Rows == weights.Rows)
                _bias = bias;
            else
                throw new ArgumentException($"{nameof(bias)}.{nameof(bias.Rows)} : {bias.Rows} doesn't correspond to {nameof(weights)}.{nameof(weights.Columns)} : {weights.Columns}");

            _weights = weights;
            InDim = _weights.Columns;
            OutDim = _weights.Rows;
            _activation = activation;
        }

        public override void Compile(IOptimizer optimizer = null)
        {
            if (InDim == 0)
                throw new InvalidOperationException("Input dimension hasn't been declared yet!");

            if (_weights == null)
            {
                // W*I + B = O
                // where W - _weights, B - _bias, I - input, O - output
                // therefore outputDimension
                _weights = new Tensor(1, OutDim, InDim, _weightInit);

                // bias isn't used for multiplication only for column-wise
                // addition, therefore names OutDim, InDim don't have any special
                // meaning for bias
                _bias = new Tensor(1, OutDim, 1, _biasInit);
            }

            optimizer?.AddLayer(_weights, _bias);
        }

        public override Tensor ForwardPass(Tensor resultPrevious, bool training = false)
        {
            lastInput = resultPrevious;
            lastOutput = _weights * resultPrevious + _bias;
            Tensor result = (_activation == null) ? lastOutput : _activation.ForwardPass(lastOutput);
            return result;
        }

        public override Tensor BackwardPass(Tensor previousGradient, out Tensor derivativeWeights, out Tensor derivativeBias)
        {
            if (lastInput == null)
                throw new InvalidOperationException("No forward pass before backward pass !");

            previousGradient = _activation == null ? previousGradient : _activation.BackwardPass(previousGradient);
            derivativeWeights = previousGradient * lastInput.Transpose();
            derivativeBias = _bias == null ? null : previousGradient;
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
