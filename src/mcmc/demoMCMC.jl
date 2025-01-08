using Distributions
using Mamba: Chains


"""
    mcmc(n_iter; Y)

Run the Gibbs algorithm to generate samples from the posterior distribution of the demo.

# Arguments :
- `n_iter::Integer`: Number of iterations.
- `Y::Vector{<:Real}`: Data.
"""
function mcmc(n_iter::Integer; Y::Vector{<:Real})

    θ = zeros(2, n_iter);
    θ[:, 1] = [0.0, 1.0];

    n = length(Y);

    for j = 2:n_iter

        θ[1, j] = rand(Normal(mean(Y), sqrt(θ[2, j-1] / n)));
        θ[2, j] = rand(InverseGamma(.5 * n, .5 * sum((Y .- θ[1, j]) .^ 2)));
        
    end

    names = [["μ"]; ["σ²"]];
    sim = Chains(copy(θ'), names=names)

    return sim

end