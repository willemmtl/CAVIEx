using Random, Distributions, SparseArrays, GMRF

"""
    generateData(grid_params, nobs)

Generate fake observations for every grid cell from a given grid of parameters.

# Arguments
- `grid_params::Array{Float64, 3}`: Concatenated grids of values for each parameter of the Normal.
- `nobs::Integer`: Number of fake observations to generate.
"""
function generateData(grid_params::Array{<:Real}, nobs::Integer)

    Y = Vector{Float64}[]

    for i = 1:size(grid_params, 1)
        for j = 1:size(grid_params, 2)
            params = grid_params[i, j, :]
            y = rand(Normal(params...), nobs)
            push!(Y, y)
        end
    end

    return Y
end


"""
    generateTargetGrid(F)

Create the grids of values for each parameter of the Normal.

For now the variance (σ²) is fixed to 1.

# Arguments
- `F::iGMRF`: Prior of the mean parameters.
"""
function generateTargetGrid(F::iGMRF)

    μ = generateParams(F);
    σ² = ones(F.G.gridSize...);

    return cat(μ, σ², dims=3)
end


"""
    generateParams(F::iGMRF)

Return a sample of a given iGMRF and reshape it to create a grid of true values for a given Normal parameter.

# Arguments
- `F::iGMRF`: The iGMRF to sample from.
"""
function generateParams(F::iGMRF)
    
    s = rand(F);
    
    return reshape(s, F.G.gridSize...)'
end