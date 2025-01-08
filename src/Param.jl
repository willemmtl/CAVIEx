using Distributions

mutable struct Param
    numero::Union{Integer, Nothing}
    approxDistribution::UnionAll
    approxMarginal::Union{Distribution, Nothing}
    mcmcSample::Union{DenseVector, Nothing}
    warmingSize::Union{Integer, Nothing}

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