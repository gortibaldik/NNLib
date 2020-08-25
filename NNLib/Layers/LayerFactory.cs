using System;

namespace NNLib.Layers
{
    public class LayerFactory
    {
        public virtual Layer CreateLayer(string name)
        {
            switch (name)
            {
                case nameof(InputLayer): return new InputLayer(1, 1, 1);
                case nameof(FlattenLayer): return new FlattenLayer();
                case nameof(DenseLayer): return new DenseLayer(1);
                default: throw new ArgumentException();
            }
        }
    }
}
