using System;
using System.Xml.Schema;
using System.Xml.Serialization;

namespace NNLib.Layers
{
    public class InputLayer : Layer, IXmlSerializable
    {
        /// <summary>
        /// Creates new InputLayer accepting input tensor of specified dimensions.
        /// </summary>
        public InputLayer(int depth, int rows, int columns)
        {
            if (rows < 0 || columns < 0 || depth < 0)
                throw new ArgumentOutOfRangeException("Dimensions of the input tensor cannot be less than or equal to zero !");

            OutDepth = depth;
            InDepth = depth;
            OutColumns = columns;
            InColumns = columns;
            OutRows = rows;
            InRows = rows;
        }

        /// <summary>
        /// Private constructor just for XML deserialization
        /// </summary>
        private InputLayer() { }

        public override Tensor BackwardPass(Tensor previousGradient, out Tensor derivativeWeights, out Tensor derivativeBias)
        {
            InputCheck(previousGradient, fwd : false);
            if (!forwardPerformed)
                throw new InvalidOperationException("No forward pass before backward pass !");
            // not trainable, output nulls
            derivativeWeights = derivativeBias = null;

            return previousGradient;
        }

        public override void Compile()
            => compiled = true;

        public override Tensor ForwardPass(Tensor input, bool training = false)
        {
            InputCheck(input);

            return input;
        }

        XmlSchema IXmlSerializable.GetSchema()
            => null;

        void IXmlSerializable.WriteXml(System.Xml.XmlWriter writer)
            // writes input and output dimensions
            => WriteXml(writer);

        void IXmlSerializable.ReadXml(System.Xml.XmlReader reader)
        {
            // reads input and output dimensions
            ReadXml(reader);
            // consumes attribute element
            reader.ReadStartElement();
        }
    }
}
