using Distributions

"""
    Param

Hold information of a CAVIEx's parameter.

# Attributes
- `numero::Union{Integer, Nothing}`: Numero of the parameter within the CAVIEx model.
- `approxDistribution::UnionAll`: Name of the marginal distribution -> must be a KNOWN name by the Distributions package.
- `approxMarginal::Union{Distribution, Nothing}`: Marginal approximating distribution of the parameter.
- `mcmcSample::Union{DenseVector, Nothing}`: .
"""
mutable struct Param
    numero::Union{Integer, Nothing}
    approxDistribution::UnionAll
    approxMarginal::Union{Distribution, Nothing}
    mcmcSample::Union{DenseVector, Nothing}
    warmingSize::Union{Integer, Nothing}

    """
    Constructor.

    # Arguments
    - `approxDistribution::UnionAll`: Name of the marginal distribution -> must be a KNOWN name by the Distributions package.
    - `num::Union{Integer, Nothing}`: Numero of the parameter within the CAVIEx model.
    """
    function Param(
        approxDistribution::UnionAll;
        num::Union{Integer, Nothing}=nothing,
    )
        
        new(num, approxDistribution, nothing, nothing, nothing);
    end

end


"""
    draw(param, N)

Draw samples of the given param from its approximating marginal.

# Arguments :
- `N::Integer`: Number of samples to draw.
"""
function draw(param::Param, N::Integer)
    return Distributions.rand(param.approxMarginal, N)
end