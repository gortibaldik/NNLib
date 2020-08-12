using System;
using NNLib.Optimizers;

namespace NNLib.Layers
{
    public abstract class Layer
    {
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

        public virtual int OutRows { get; protected set; } = -1;
        public virtual int OutColumns { get; protected set; } = -1;
        public virtual int OutDepth { get; protected set; } = -1;

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
