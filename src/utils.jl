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
    adaptHp!(dist, values)

Bridge between our usual notation of law's parameters and the one in Distributions.jl.

# Arguments :
- `dist::UnionAll`: A distribution's name.
- `values::DenseVector`: The values of the parameters as we usually write them.
"""
function adaptHp!(dist::UnionAll, values::DenseVector)

    if dist == Normal
        values[2] = sqrt(values[2]);
    elseif dist == Gamma
        values[2] = 1 / values[2];
    end

end