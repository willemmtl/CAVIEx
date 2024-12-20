using GMRF

if !isdefined(Main, :AbstractModel)
    include("../models/AbstractModel.jl");
end
using .AbstractModel
if !isdefined(Main, :Demo)
    include("../models/Demo.jl");
end
using .Demo

include("../dataGen/dataGenNormal.jl");


"""
    InstanceDemo

Generate an instance of a Demo model.

# Attributes :
- `realParams::DenseVector`: True values of μ and σ².
- `data::Vector{Vector{<:Real}}`: Sample from the real distribution.
- `model::AbstractModel.BaseModel`: Every function needed in the model.
    -> See BaseModel.
"""
struct InstanceDemo
    realParams::DenseVector
    data::Vector{Vector{<:Real}}
    model::AbstractModel.BaseModel

    function InstanceDemo(;
        seed::Integer,
        realmu::Float64,
        realsigma2::Float64,
    )
        Random.seed!(seed);
        nobs = 100;
        data = [rand(Normal(realmu, sqrt(realsigma2)), nobs)];

        hyperParams = ["α", "β", "m", "s²"];

        new(
            [realmu, realsigma2],
            data,
            AbstractModel.BaseModel(
                hyperParams,
                θ -> Demo.logTargetDensity(θ, Y=data),
                (θ, Hθ) -> Demo.logApproxDensity(θ, Hθ),
                [
                    Hθ -> Demo.μMarginal(Hθ),
                    Hθ -> Demo.σ²Marginal(Hθ),
                ],
                [
                    Hθ -> Demo.refine_α(Y=data),
                    Hθ -> Demo.refine_β(Hθ, Y=data),
                    Hθ -> Demo.refine_m(Y=data),
                    Hθ -> Demo.refine_s²(Hθ, Y=data),
                ]
            )
        )
    end

end