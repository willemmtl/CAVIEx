using GMRF

if !isdefined(Main, :AbstractModel)
    include("../models/AbstractModel.jl");
end
using .AbstractModel
if !isdefined(Main, :NormalMeanField)
    include("../models/NormalMeanField.jl");
end
using .NormalMeanField

include("../dataGen/dataGenNormal.jl");


"""
    InstanceNMF

Generate an instance of a NormalMeanField model.

# Attributes :
- `F::iGMRF`: Spatial scheme.
- `gridTarget::Array{Float64, 3}`: True values of μ.
- `data::Vector{Vector{<:Real}}`: Extreme values for each cell.
- `model::AbstractModel.BaseModel`: Every function needed in the model.
    -> See BaseModel.
"""
struct InstanceNMF
    F::iGMRF
    gridTarget::Array{Float64, 3}
    data::Vector{Vector{<:Real}}
    model::AbstractModel.BaseModel

    function InstanceNMF(
        m₁::Integer,
        m₂::Integer;
        seed::Integer,
        biasmu::Float64,
        realkappa::Float64,
    )
        Random.seed!(seed);
        F = iGMRF(m₁, m₂, 1, realkappa);
        gridTarget = generateTargetGrid(F);
        gridTarget[:, :, 1] = gridTarget[:, :, 1] .+ biasmu;
        nobs = 100;
        data = generateData(gridTarget, nobs);

        m = m₁ * m₂;
        hyperParams = [["η$k" for k = 1:m]..., ["s²$k" for k = 1:m]..., "aᵤ", "bᵤ"];

        new(
            F,
            gridTarget,
            data,
            AbstractModel.BaseModel(
                hyperParams,
                θ -> NormalMeanField.logTargetDensity(θ, F=F, Y=data),
                (θ, Hθ) -> NormalMeanField.logApproxDensity(θ, Hθ, F=F),
                [
                    [Hθ -> NormalMeanField.μMarginal(Hθ, k=k, F=F) for k = 1:m]...,
                    Hθ -> NormalMeanField.κMarginal(Hθ, F=F)
                ],
                [
                    [Hθ -> NormalMeanField.refine_η(Hθ, k, F=F, Y=data) for k = 1:m]...,
                    [Hθ -> NormalMeanField.refine_s²(Hθ, k, F=F, Y=data) for k = 1:m]...,
                    Hθ -> NormalMeanField.refine_aᵤ(F=F),
                    Hθ -> NormalMeanField.refine_bᵤ(Hθ, F=F),
                ]
            )
        )
    end

end