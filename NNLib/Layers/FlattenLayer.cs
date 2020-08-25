using NNLib.Optimizers;
using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;

namespace NNLib.Layers
{
    public class FlattenLayer : Layer, IXmlSerializable
    {
        public override int OutDepth { get => 1; }
        public override int OutColumns { get => 1; }
        public override int OutRows { get => InRows.Value * InColumns.Value * InDepth.Value; }

        private int lastBatchSize;

        public FlattenLayer()
        {
            InDepth = InRows = InColumns = null;
        }

        public FlattenLayer(int inDepth, int inRows, int inColumns) : this()
        {
            if (inRows < 0 || inColumns < 0 || inDepth < 0)
                throw new ArgumentOutOfRangeException("Dimensions of the tensor cannot be less than or equal to zero !");

            InDepth = inDepth;
            InRows = inRows;
            InColumns = inColumns;
        }

        public override Tensor BackwardPass(Tensor previousGradient, out Tensor derivativeWeights, out Tensor derivativeBias)
        {
            InputCheck(input: previousGradient, fwd: false);
            if (!forwardPerformed)
                throw new InvalidOperationException("No forward pass before backward pass !");
                
            // not trainable, output nulls
            derivativeWeights = derivativeBias = null;
            forwardPerformed = false;

            return previousGradient.Reshape(lastBatchSize, InDepth.Value, InRows.Value, InColumns.Value);
        }

        public override void Compile()
            => compiled = true;

        public override Tensor ForwardPass(Tensor input, bool training = false)
        {
            InputCheck(input);

            forwardPerformed = true;
            lastBatchSize = input.BatchSize;
            return input.Reshape(lastBatchSize, OutDepth, OutRows, OutColumns);
        }

        XmlSchema IXmlSerializable.GetSchema()
            => null;

        void IXmlSerializable.ReadXml(XmlReader reader)
        {
            // reads input and output dimensions
            ReadXml(reader);
            // consumes attribute element
            reader.ReadStartElement();
        }

        void IXmlSerializable.WriteXml(XmlWriter writer)
            // writes input and output dimensions
            => WriteXml(writer);
    }
}
