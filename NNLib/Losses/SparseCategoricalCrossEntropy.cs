using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib.Losses
{
    public class SparseCategoricalCrossEntropy : ILossLayer
    {
        public Tensor BackwardPass()
        {
            throw new NotImplementedException();
        }

        public double ForwardPass(Tensor neuralOutput, Tensor expectedOutput)
        {
            throw new NotImplementedException();
        }
    }
}
