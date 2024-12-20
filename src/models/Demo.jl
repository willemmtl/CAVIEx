module Demo

using Statistics

"""
    logTargetDensity(θ; Y)

Log target density : posterior of Normal likelihood + Jeffreys priors.

# Arguments :
- `θ::DenseVector`: Parameters to infer [μ, σ²].
- `Y::Vector{<:Real}`: Sample.
"""
function logTargetDensity(θ::DenseVector; Y::Vector{<:Real})
    
    n = length(Y)
    
    return - (0.5 * n + 1) * log(θ[2]) - sum((Y .- θ[1]).^2) / (2*θ[2])
end


"""
    logApproxDensity(θ, Hθ)

# Arguments :
- `θ::DenseVector`: Parameters to infer [μ, σ²].
- `Hθ::Vector{<:Real}`: Hyper-parameters [α, β, m, s²].
"""
function logApproxDensity(θ::DenseVector, Hθ::Vector{<:Real})
    return logpdf(μMarginal(Hθ), θ[1]) + logpdf(σ²Marginal(Hθ), θ[2])
end


"""
    μMarginal(Hθ)

# Arguments :
- `Hθ::DenseVector`: Hyper-parameters -> [α, β, m, s²].
"""
function μMarginal(Hθ::DenseVector)
    return Normal(Hθ[3], sqrt(Hθ[4]))
end


"""
    σ²Marginal(Hθ)

# Arguments :
- `Hθ::DenseVector`: Hyper-parameters -> [α, β, m, s²].
"""
function σ²Marginal(Hθ::DenseVector)
    return InverseGamma(Hθ[1], Hθ[2])
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
- `Hθ::DenseVector`: Hyper-parameters -> [α, β, m, s²].
- `Y::Vector{Float64}`: Sample.
"""
function refine_β(Hθ::DenseVector; Y::Vector{Float64})
    
    m = Hθ[3];
    s² = Hθ[4];

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
- `Hθ::DenseVector`: Hyper-parameters -> [α, β, m, s²].
"""
function refine_s²(Hθ::DenseVector; Y::Vector{Float64})
    
    α = Hθ[1];
    β = Hθ[2];

    return (α + β + 2) / length(Y) / (α + 2)
end

end