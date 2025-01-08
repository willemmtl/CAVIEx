"""
    HyperParam

Hold information of a CAVIEx's hyper-parameter.

# Attributes
- `numero::Union{Integer, Nothing}`: Numero of the hyper-parameter within the CAVIEx model.
- `refiningFunction::Function`: Function to call to update the parameter.
- `trace::Union{Vector{Float64}, Nothing}`: All the values the hyper-parameter has had during the CAVI algorithm.
"""
mutable struct HyperParam
    numero::Union{Integer, Nothing}
    refiningFunction::Function
    trace::Union{Vector{Float64}, Nothing}

    """
    Constructor.

    # Arguments
    - `refiningFunction::Function`: Function to call to update the parameter.
    - `num::Union{Integer, Nothing}`: Numero of the hyper-parameter within the CAVIEx model.
    """
    function HyperParam(
        refiningFunction::Function;
        num::Union{Integer, Nothing}=nothing,
    )
        return new(num, refiningFunction, nothing)
    end
end


"""
    current(hp)

Return the current value of a given hyper-parameter.
Throw error if the trace is empty.
"""
function current(hp::HyperParam)
    if !isnothing(hp.trace)
        return hp.trace[end]
    else
        error("Hyper-parameter is empty !")
    end
end