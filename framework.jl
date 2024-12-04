using Distributions, Optim, ForwardDiff

# """
#     dataLevelLogLike(Y, θ)

# Log-likelihood at the data level for our given hierarchical model.

# Will be useful to compute the location parameters' Fisher Information.

# # Arguments :
# - `Y::Vector{Vector{Float64}}`: Extreme data for each cell.
# - `μ::DenseVector`: Location parameters.
# """
# function dataLevelLogLike(Y::Vector{Vector{Float64}}, μ::DenseVector)
#     return sum(loglikelihood.(Gumbel.(μ, 1), Y))
# end


function logFunctionalFormPosterior(θ::DenseVector; F::iGMRF, Y::Vector{Vector{Float64}})
    return (
        sum(loglikelihood.(Gumbel.(θ[2:end], 1), Y)) 
        + ((length(θ)-1) - F.r) * log(θ[1]) / 2 
        - θ[1] * θ[2:end]' * F.G.W * θ[2:end] / 2
        - θ[1] / 100
    )
end;


function findMode(f::Function, θ₀::DenseVector)
    F(θ::DenseVector) = -f(θ)
    return optimize(F, θ₀, Newton(); autodiff = :forward)
end;


function computeFisherVariance(logf::Function, θ̂::AbstractArray)
    return inv(computeFisherInformation(θ -> logf(θ), θ̂))
end


function computeFisherInformation(logf::Function, θ̂::AbstractArray)
    return - ForwardDiff.hessian(logf, θ̂)
end;