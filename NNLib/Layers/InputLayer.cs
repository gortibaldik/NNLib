using System;
using System.Xml.Schema;
using System.Xml.Serialization;

namespace NNLib.Layers
{
    public class InputLayer : Layer, IXmlSerializable
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

        /// <summary>
        /// Private constructor just for XML deserialization
        /// </summary>
        private InputLayer() { }

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

        XmlSchema IXmlSerializable.GetSchema()
            => null;

        void IXmlSerializable.WriteXml(System.Xml.XmlWriter writer)
            => WriteXml(writer);

        void IXmlSerializable.ReadXml(System.Xml.XmlReader reader)
        {
            ReadXml(reader);
            reader.ReadStartElement();
        }
    }
}
