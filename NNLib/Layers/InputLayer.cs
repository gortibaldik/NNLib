using NNLib.Optimizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib.Layers
{
    public class InputLayer : Layer
    {
        public InputLayer(int depth, int rows, int columns)
        {
            OutDepth = depth;
            InDepth = depth;
            OutColumns = columns;
            InColumns = columns;
            OutRows = rows;
            InRows = rows;
        }

        public override Tensor BackwardPass(Tensor previousGradient, out Tensor derivativeWeights, out Tensor derivativeBias)
        {
            derivativeWeights = null;
            derivativeBias = null;
            return previousGradient;
        }

        public override void Compile()
            => compiled = true;

        public override Tensor ForwardPass(Tensor input, bool training = false)
        {
            InputCheck(input);

            return input;
        }
    }
}
