using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib.Losses
{
    /// <summary>
    /// Implementation cannot be considered as complete or even partially complete. 
    /// Only one supported mode, with SparseCategoricalCrossEntropy directly following
    /// the softmax
    /// </summary>
    public class SparseCategoricalCrossEntropy : ILossLayer
    {
        private Tensor neuralOutput;
        private Tensor expectedOutput;

        /// <summary>
        /// Only one supported mode, with SparseCategoricalCrossEntropy directly following
        /// the softmax. Hence the result of the BackwardPass isn't the gradient of the 
        /// SparseCategoricalCrossEntropy with respect to softmax(neuralOutput).
        /// The result is the gradient of SparseCategoricalCrossEntropy(softmax(neuralOutput))
        /// with respect to neuralOutput.
        /// </summary>
        public Tensor BackwardPass()
        {
            if (neuralOutput == null || expectedOutput == null)
                throw new InvalidOperationException("Backward pass before forward pass exception !");

            var result = neuralOutput.ApplyFunctionOnAllElements((actual, expected) => actual - expected, expectedOutput, disableChecking: true);
            neuralOutput = expectedOutput = null;
            return result;
        }

        public double ForwardPass(Tensor neuralOutput, Tensor expectedOutput)
        {
            if (neuralOutput.Rows != expectedOutput.Rows || neuralOutput.Columns != expectedOutput.Columns || neuralOutput.Depth != expectedOutput.Depth
                || neuralOutput.BatchSize != expectedOutput.BatchSize)
                throw new InvalidOperationException("The output of the neural net doesn't correspond to the expected output!");

            this.neuralOutput = neuralOutput;
            this.expectedOutput = expectedOutput;
            var res = 0D;

            neuralOutput.ApplyFunctionOnAllElements((actual, expected) => { res -= expected * Math.Log(actual); return actual; }, expectedOutput, disableChecking : true);

            return res / neuralOutput.BatchSize;
        }
    }
}
