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
    logApproxDensity(θ; Hθ)

# Arguments :
- `θ::DenseVector`: Parameters to infer [μ, σ²].
- `Hθ::Vector{<:Real}`: Hyper-parameters [α, β, m, s²].
"""
function logApproxDensity(θ::DenseVector; Hθ::Vector{<:Real})
    return logpdf(Normal(Hθ[3], sqrt(Hθ[4])), θ[1]) + logpdf(Gamma(Hθ[1], Hθ[2]), θ[2])
end


"""
    compute_α(; Y)

Compute the first parameter of σ²'s approximation.

# Arguments :
- `Y::Vector{Float64}`:Sample.
"""
function compute_α(; Y::Vector{Float64})
    return length(Y)/2
end


"""
    compute_β(m, s²; Y)

Compute the second parameter of σ²'s approximation.

# Arguments :
- `m::Real`: Mean of μ's approximation.
- `s²::Real`: Variance of μ's approximation.
- `Y::Vector{Float64}`: Sample.
"""
function compute_β(m::Real, s²::Real; Y::Vector{Float64})
    return 0.5 * (sum(Y.^2) + length(Y)*(s² + m^2) - 2*m*length(Y)*mean(Y))
end


"""
    compute_m(; Y)

Compute the mean of μ's approximation.

# Arguments :
- `Y::Vector{Float64}`: Sample.
"""
function compute_m(; Y::Vector{Float64})
    return mean(Y)
end


"""
    compute_s²(α, β; Y)

Compute the variance of μ's approximation.

# Arguments :
"""
function compute_s²(α::Real, β::Real; Y::Vector{Float64})
    return (α + β + 2) / length(Y) / (α + 2)
end