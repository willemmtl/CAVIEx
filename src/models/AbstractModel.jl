module AbstractModel

using Distributions

abstract type Model end

"""
    BaseModel

Sets the context for the variational algorithm.

# Attributes :
- `nHyperParams::Integer`: Number of hyper-parameters of the approximating function.
- `Hθ::DenseVector`: Hyper-parameters.
- `logTargetDensity::Function`: Density to approximate (log).
    MUST have the shape θ -> f(θ) where θ is the vectors of parameters.
- `logApproxDensity::Function`: Approximating density (log).
    MUST have the shape θ, Hθ -> f(θ, Hθ) where θ and Hθ are respectively the vector of parameters and hyper-parameters.
- `approxMarginals::Vector{Function}`: Approximating marginal distribution for each hyper-parameter.
    MUST have the shape f(Hθ) where Hθ is the vector of hyper-parameters.
- `refiningFunctions::Vector{Function}`: Functions used to update hyper-parameters in the CAVI algorithm.
    MUST have the shape (Hθ) -> f(Hθ) where Hθ is the vector of hyper-parameters.
"""
struct BaseModel
    hyperParams::Vector{<:String}
    nHyperParams::Integer
    hyperParamsValue::Vector{Union{Float64, Nothing}}
    hyperParamsTrace::Dict{String, Union{Vector{Float64}, Nothing}}
    logTargetDensity::Function
    logApproxDensity::Function
    approxMarginals::Vector{<:Function}
    refiningFunctions::Vector{<:Function}

    function BaseModel(
        hyperParams::Vector{<:String},
        logTargetDensity::Function,
        logApproxDensity::Function,
        approxMarginals::Vector{<:Function},
        refiningFunctions::Vector{<:Function},
    )
        nHyperParams = length(hyperParams)
        hyperParamsValue = [nothing for _ = 1:nHyperParams];
        hyperParamsTrace = Dict(hyperParams .=> nothing);
        new(
            hyperParams,
            nHyperParams,
            hyperParamsValue,
            hyperParamsTrace,
            logTargetDensity,
            logApproxDensity,
            approxMarginals,
            refiningFunctions,
        )
    end

end

"""
    storeValues(model)

Stores the hyper-parameters' current value to build the trace.
"""
function storeValues(model::BaseModel)

    for (k, hyperParam) in enumerate(model.hyperParams)
        if !isnothing(model.hyperParamsValue[k])
            if !isnothing(model.hyperParamsTrace[hyperParam])
                push!(model.hyperParamsTrace[hyperParam], model.hyperParamsValue[k]);
            else
                model.hyperParamsTrace[hyperParam] = [model.hyperParamsValue[k]];
            end
        else
            error("$hyperParam does not have any value !");
        end
    end

end

end