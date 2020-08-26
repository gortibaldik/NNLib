using System;
using System.Data;
using System.IO;
using NNLib.Optimizers;

namespace NNLib.Layers
{
    public abstract class Layer
    {
        /// <summary>
        /// Rows dimension of the tensor returned from Layer.Forward call
        /// </summary>
        public virtual int OutRows { get; protected set; } = -1;

        /// <summary>
        /// Columns dimension of the tensor returned from Layer.Forward call
        /// </summary>
        public virtual int OutColumns { get; protected set; } = -1;

        /// <summary>
        /// Depth dimension of the tensor returned from Layer.Forward call
        /// </summary>
        public virtual int OutDepth { get; protected set; } = -1;

        private int? inRows;

        /// <summary>
        /// Expected Rows dimension of the input tensor
        /// </summary>
        public int? InRows 
        { 
            get => inRows; 
            internal set => inRows = compiled == false ? value : throw new InvalidOperationException("Cannot change input dimensions of already compiled layer !");
        }

        private int? inColumns;

        /// <summary>
        /// Expected Columns dimension of the input tensor
        /// </summary>
        public int? InColumns 
        { 
            get => inColumns; 
            internal set => inColumns = compiled == false ? value : throw new InvalidOperationException("Cannot change input dimensions of already compiled layer !");
        }

        private int? inDepth;

        /// <summary>
        /// Expected Depth dimension of the input tensor
        /// </summary>
        public int? InDepth 
        { 
            get => inDepth; 
            internal set => inDepth = compiled == false ? value : throw new InvalidOperationException("Cannot change input dimensions of already compiled layer !");
        }

        protected bool forwardPerformed = true;
        protected bool compiled = false;


        /// <summary>
        /// Forward traversal through the layer, calculates output based on the input tensor.
        /// May cache the input in order to be prepared for the backward pass.
        /// Layer must be compiled in order to be able to perform the forward pass.
        /// </summary>
        public abstract Tensor ForwardPass(Tensor input);

        /// <summary>
        /// Prepares the layer for the usage, initializes the weights if needed. Layer shouldn't
        /// be usable without being compiled and should check if the compilation was already performed.
        /// </summary>
        public abstract void Compile();

        /// <summary>
        /// Backward traversal through the layer, calculates the output based on the input tensor.
        /// 
        /// Returns the gradient in respect to last input and returns gradients with respect to weights
        /// and biases (the out parameters gradients should be averages across all the batches).
        /// If layer doesn't have weights/bias structure, out parameters returned have to be
        /// just null values.
        ///
        /// ForwardPass must be performed before BackwardPass. If not InvalidOperationException should
        /// be thrown.
        /// Layer must be compiled in order to be able to perform the backward pass.
        /// </summary>
        public abstract Tensor BackwardPass(Tensor previousGradient, out Tensor derivativeWeights, out Tensor derivativeBias);


        /// <summary>
        /// Checks if the layer is compiled, based on bool fwd (defaults to true) checks if
        /// the input tensor matches the input (output if fwd is false) dimensions of the 
        /// layer.
        /// </summary>
        protected void InputCheck(Tensor input, bool fwd = true)
        {
            if (!compiled)
                throw new InvalidOperationException("Cannot work with uncompiled network !");

            if (fwd)
            {
                if (input.Depth != InDepth || input.Rows != InRows || input.Columns != InColumns)
                    throw new FormatException($"Not valid shape of a tensor ! : EXPECTED : {InDepth}x{InRows}x{InColumns} GOT : {input.Depth}x{input.Rows}x{input.Columns}");
            }
            else if (input.Depth != OutDepth || input.Rows != OutRows || input.Columns != OutColumns)
                throw new FormatException($"Not valid shape of a tensor ! : EXPECTED : {OutDepth}x{OutRows}x{OutColumns} GOT : {input.Depth}x{input.Rows}x{input.Columns}");
        }

        /// <summary>
        /// Abstract object layer is not serializable, therefore this is intended to be used as
        /// a prolog to xml serialization - serializing input and output dimensions.
        /// </summary>
        protected void WriteXml(System.Xml.XmlWriter writer)
        {
            if (inRows == null)
                throw new InvalidOperationException($"Cannot serialize uninitialized layer, {nameof(inRows)} == null !");

            if (inColumns == null)
                throw new InvalidOperationException($"Cannot serialize uninitialized layer, {nameof(inColumns)} == null !");

            if (inDepth == null)
                throw new InvalidOperationException($"Cannot serialize uninitialized layer, {nameof(inDepth)} == null !");

            writer.WriteAttributeString("InDims", "[" + InRows.ToString() + "," + InColumns.ToString() + "," + InDepth.ToString() + "]");
            writer.WriteAttributeString("OutDims", "[" + OutRows.ToString() + "," + OutColumns.ToString() + "," + OutDepth.ToString() + "]");
        }

        /// <summary>
        /// Default parser of dimensions from the format "[rows,columns,depth]"
        /// </summary>
        private void Parse(string toBeParsed, out int? rows, out int? columns, out int? depth)
        {
            var stringReader = new StringReader(toBeParsed);
            var numbers = new int[3];
            int r;
            stringReader.Read();
            for (int i = 0; i < 3; i++)
            {
                while ((r = stringReader.Read()) != ']' && r != ',')
                {
                    if (r >= '0' && r <= '9')
                        numbers[i] = numbers[i] * 10 + r - '0';
                    else
                        throw new DataMisalignedException();
                }
            }
            rows = numbers[0];
            columns = numbers[1];
            depth = numbers[2];
        }

        /// <summary>
        /// Abstract object layer is not serializable, therefore this is intended to be used as
        /// a prolog to xml deserialization - deserializing input and output dimensions.
        /// </summary>
        protected void ReadXml(System.Xml.XmlReader reader)
        {
            reader.MoveToContent();
            string inDims = reader.GetAttribute("InDims");
            string outDims = reader.GetAttribute("OutDims");

            Parse(inDims, out inRows, out inColumns, out inDepth);
            Parse(outDims, out int? rows, out int? columns, out int? depth);

            OutRows = rows.Value;
            OutColumns = columns.Value;
            OutDepth = depth.Value;
            compiled = true;
        }
    }

    public struct Shape
    {
        public int Depth { get; }
        public int Rows { get; }
        public int Columns { get; }

        /// <summary>
        /// Creates new Shape structure, and performs non-negativity checks
        /// on all the parameters.
        /// </summary>
        public Shape(int depth, int rows, int columns)
        {
            Depth = depth > 0 ? depth : throw new ArgumentOutOfRangeException();
            Rows = rows > 0 ? rows : throw new ArgumentOutOfRangeException();
            Columns = columns > 0 ? columns : throw new ArgumentOutOfRangeException();
        }
    }
}
