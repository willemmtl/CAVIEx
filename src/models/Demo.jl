"""
Demonstration Model.

The data are X1, ..., Xn drawn from Normal(μ, σ²)
The target density is the posterior of θ = [μ, σ²].
The mean-field aproximation gives
    μ ∼ Normal(m, s²)
    σ² ∼ InverseGamma(α, β)
where [m, s², α, β] are the hyper-parameters.
"""

module Demo

using Statistics
include("../mcmc/demoMCMC.jl");


"""
    logTargetDensity(θ; Y)

Log target density : posterior of Normal likelihood + Jeffreys priors.

# Arguments : TBD
"""
function logTargetDensity(θ::DenseVector; Y::Vector{<:Real})
    
    n = length(Y);
    
    return - (0.5 * n + 1) * log(θ[2]) - sum((Y .- θ[1]).^2) / (2*θ[2])
end


"""
    logApproxDensity(θ, Hθ)

# Arguments : TBD
"""
function logApproxDensity(θ::DenseVector, Hθ::DenseVector)
    return logpdf(Normal(Hθ[1], sqrt(Hθ[2])), θ[1]) + logpdf(InverseGamma(Hθ[3], Hθ[4]), θ[2])
end


"""
    refine_α(; Y)

Compute the first parameter of σ²'s approximation.

# Arguments :
- `Y::Vector{Float64}`:Sample.
"""
function refine_α(; Y::Vector{Float64})
    return length(Y)/2
end


"""
    refine_β(Hθ; Y)

Compute the second parameter of σ²'s approximation.

# Arguments :
- `Hθ::DenseVector`: Hyper-parameters -> [m, s², α, β].
- `Y::Vector{Float64}`: Sample.
"""
function refine_β(Hθ::DenseVector; Y::Vector{Float64})
    
    m = Hθ[1];
    s² = Hθ[2];

    return 0.5 * (sum(Y.^2) + length(Y)*(s² + m^2) - 2*m*length(Y)*mean(Y))
end


"""
    refine_m(; Y)

Compute the mean of μ's approximation.

# Arguments :
- `Y::Vector{Float64}`: Sample.
"""
function refine_m(; Y::Vector{Float64})
    return mean(Y)
end


"""
    refine_s²(Hθ; Y)

Compute the variance of μ's approximation.

# Arguments :
- `Hθ::DenseVector`: Hyper-parameters -> [m, s², α, β].
"""
function refine_s²(Hθ::DenseVector; Y::Vector{Float64})
    
    α = Hθ[3];
    β = Hθ[4];

    return (α + β + 2) / length(Y) / (α + 2)
end

end