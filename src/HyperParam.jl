mutable struct HyperParam
    numero::Union{Integer, Nothing}
    refiningFunction::Function
    trace::Union{Vector{Float64}, Nothing}

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