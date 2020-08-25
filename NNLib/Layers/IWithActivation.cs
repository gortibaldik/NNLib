using NNLib.Activations;

namespace NNLib.Layers
{
    interface IWithActivation
    {
        public string ActivationUsed { get; }
    }
}
