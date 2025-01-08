using Distributions, Random, OrderedCollections

if !isdefined(Main, :CAVIEx)
    include("../CAVIEx.jl");
end
using .CAVIEx

if !isdefined(Main, :Demo)
    include("../models/Demo.jl");
end
using .Demo

"""
    InstanceDemo

Generate an instance of a Demo model.

# Attributes :
- `realParams::DenseVector`: True values of μ and σ².
- `data::Vector{Vector{<:Real}}`: Sample from the real distribution.
- `model::CAVIEx.Model`: Generic model.
"""
struct InstanceDemo
    realParams::Dict{Symbol, Float64}
    data::Vector{<:Real}
    model::CAVIEx.Model

    function InstanceDemo(;
        seed::Integer,
        realmu::Float64,
        realsigma2::Float64,
    )
        Random.seed!(seed);
        nobs = 100;
        data = Distributions.rand(Normal(realmu, sqrt(realsigma2)), nobs);

        params = OrderedDict(
            :μ => (
                Normal,
                OrderedDict(
                    :μ => (1, :m, Hθ -> Demo.refine_m(Y=data)),
                    :σ => (2, :s², Hθ -> Demo.refine_s²(Hθ, Y=data)),
                )
            ),
            :σ² => (
                InverseGamma,
                OrderedDict(
                    :invd => (3, :α, Hθ -> Demo.refine_α(Y=data)),
                    :θ => (4, :β, Hθ -> Demo.refine_β(Hθ, Y=data)),
                )
            ),
        );

        new(
            Dict(:realmu => realmu, :realsigma2 => realsigma2),
            data,
            CAVIEx.Model(
                params,
                θ -> Demo.logTargetDensity(θ, Y=data),
                (θ, Hθ) -> Demo.logApproxDensity(θ, Hθ),
                niter -> Demo.mcmc(niter, Y=data),
            ),
        )
    end

end