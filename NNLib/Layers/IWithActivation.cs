using NNLib.Activations;

namespace NNLib.Layers
{
    interface IWithActivation
    {
        public ActivationFunctions ActivationUsed { get; }
    }
}
