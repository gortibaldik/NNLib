using System;
using System.Data;
using System.IO;
using NNLib.Optimizers;

namespace NNLib.Layers
{
    public abstract class Layer
    {
        public virtual int OutRows { get; protected set; } = -1;
        public virtual int OutColumns { get; protected set; } = -1;
        public virtual int OutDepth { get; protected set; } = -1;

        private int? inRows;

        /// <summary>
        /// Input.Rows dimension
        /// </summary>
        public int? InRows 
        { 
            get => inRows; 
            internal set => inRows = compiled == false ? value : throw new InvalidOperationException("Cannot change input dimensions of already compiled layer !");
        }

        private int? inColumns;

        /// <summary>
        /// Input.Columns dimension
        /// </summary>
        public int? InColumns 
        { 
            get => inColumns; 
            internal set => inColumns = compiled == false ? value : throw new InvalidOperationException("Cannot change input dimensions of already compiled layer !");
        }

        private int? inDepth;

        /// <summary>
        /// Input.Depth dimension
        /// </summary>
        public int? InDepth 
        { 
            get => inDepth; 
            internal set => inDepth = compiled == false ? value : throw new InvalidOperationException("Cannot change input dimensions of already compiled layer !");
        }

        protected bool compiled = false;


        public abstract Tensor ForwardPass(Tensor input, bool training = false);

        public abstract void Compile();

        public abstract Tensor BackwardPass(Tensor previousGradient, out Tensor derivativeWeights, out Tensor derivativeBias);

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

        protected void ReadXml(System.Xml.XmlReader reader)
        {
            reader.MoveToContent();
            string inDims = reader.GetAttribute("InDims");
            string outDims = reader.GetAttribute("OutDims");
            int? rows, columns, depth;

            Parse(inDims, out inRows, out inColumns, out inDepth);
            Parse(outDims, out rows, out columns, out depth);

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

        public Shape(int depth, int rows, int columns)
        {
            Depth = depth;
            Rows = rows;
            Columns = columns;
        }
    }
}
