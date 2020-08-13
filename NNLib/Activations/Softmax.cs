using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace NNLib.Activations
{
    /// <summary>
    /// Only one supported mode, with 1xNx1 tensor shape at the input and softmax right before CrossEntropy Loss
    /// </summary>
    public class Softmax : IActivationLayer
    {
        public ActivationFunctions Name { get => ActivationFunctions.Softmax; }

        /// <summary>
        /// This isn't real and shouldn't be considered as a real implementation of the softmax activation. 
        /// Neural network itself will check if 
        /// </summary>
        /// <param name="previousGradient"></param>
        /// <returns></returns>
        public Tensor BackwardPass(Tensor previousGradient)
            => previousGradient;

        public Tensor BackwardPassFromCrossEntropy(Tensor previousGradient, bool alreadyComputed = true)
            => alreadyComputed ? previousGradient : BackwardPass(previousGradient);

        public Tensor ForwardPass(Tensor input)
        {
            // not numerically stable!
            if (input.Depth != 1 || input.Columns != 1)
                throw new NotImplementedException("The softmax layer doesn't support other than 1 x N x 1 shape of the tensor!");

            var sum = 0D;
            var max = double.MinValue;
            input.ApplyFunctionOnAllElements(x => 
            { 
                if (x > max) 
                    max = x;
                return x;
                });
            var exponentiated = input.ApplyFunctionOnAllElements(x => { var d = Math.Exp(x-max); sum += d; return d; });
            var softmax = exponentiated.ApplyFunctionOnAllElements(x => x / sum);

            return softmax;
        }
    }
}
