using NNLib.Optimizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib.Layers
{
    public class FlattenLayer : Layer
    {
        public override int OutDepth { get => 1; }
        public override int OutColumns { get => 1; }
        public override int OutRows { get => InRows.Value * InColumns.Value * InDepth.Value; }

        private int lastBatchSize;

        public FlattenLayer(int inDepth, int inRows, int inColumns)
        {
            InDepth = inDepth;
            InRows = inRows;
            InColumns = inColumns;
        }

        public FlattenLayer()
        {
            InDepth = InRows = InColumns = null;
        }

        public override Tensor BackwardPass(Tensor previousGradient, out Tensor derivativeWeights, out Tensor derivativeBias)
        {
            InputCheck(input: previousGradient, fwd: false);

            derivativeWeights = derivativeBias = null;
            return previousGradient.Reshape(lastBatchSize, InDepth.Value, InRows.Value, InColumns.Value);
        }

        public override void Compile()
            => compiled = true;

        public override Tensor ForwardPass(Tensor input, bool training = false)
        {
            InputCheck(input);

            lastBatchSize = input.BatchSize;
            return input.Reshape(lastBatchSize, OutDepth, OutRows, OutColumns);
        }
    }
}
