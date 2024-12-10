using Distributions, Optim, ForwardDiff, LinearAlgebra


"""
    evaluateLogMvDensity(f, supp)

Evaluate a log multivariate density over a given set of vectors.

# Arguments :
- `f::Function`: Log multivariate density function to evaluate.
- `supp::Matrix{<:Real}`: Set of n p-arrays stored in a (p x n) matrix.
"""
function evaluateLogMvDensity(f::Function, supp::Matrix{<:Real})
    return vec(mapslices(f, supp, dims=1))
end


"""
    findMode(f, θ₀)

Given a density's functional form, find a mode.

Warning : there may be several modes.
This function will catch the first one it finds depending on the initialization.

# Arguments :
- `f::Function`: A density's functional form (can be log).
- `θ₀::DenseVector`: Parameters' initialization.
"""
function findMode(f::Function, θ₀::DenseVector)
    F(θ::DenseVector) = -f(θ)
    return optimize(F, θ₀, Newton(); autodiff = :forward)
end;


"""
    computeFisherVariance(logf, θ̂)

Compute the quadratic approximation variance matrix of a given distribution.
It is based on the Fisher information.

The link is Σ = F⁻¹ where F is the Fisher information matrix.

# Arguments :
- `logf::Function`: log the density's functional form.
- `θ̂::AbstractArray`: mode of the distribution.
"""
function computeFisherVariance(logf::Function, θ̂::AbstractArray)
    return inv(computeFisherInformation(θ -> logf(θ), θ̂))
end


"""
    computeFisherInformation(logf, θ̂)

Compute the Fisher information matrix of a given distribution.

# Arguments :
- `logf::Function`: log the density's functional form.
- `θ̂::AbstractArray`: mode of the distribution.
"""
function computeFisherInformation(logf::Function, θ̂::AbstractArray)
    return - ForwardDiff.hessian(logf, θ̂)
end;