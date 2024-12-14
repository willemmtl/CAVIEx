using Test

include("../../src/models/AbstractModel.jl");
using .AbstractModel
include("../../src/models/PrecipMeanField.jl");
using .PrecipMeanField

include("../../src/iGMRF.jl");

@testset "AbstractModel.jl" begin
    
    @testset "BaseModel(Hθ₀, logTargetDensity, logApproxMarginals, refiningFunctions)" begin
        
        F = PrecipMeanField.iGMRF(2, 2, 1);
        Y = [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ];

        Hθ₀ = [zeros(4)..., ones(4)..., 1.0, 1.0]

        precipModel = AbstractModel.BaseModel(
            Hθ₀,
            θ -> PrecipMeanField.logTargetDensity(θ, F=F, Y=Y),
            [
                [(θ, Hθ) -> PrecipMeanField.μMarginal(θ, Hθ, k=k, F=F) for k = 1:4]...,
                (θ, Hθ) -> PrecipMeanField.κMarginal(θ, Hθ, F=F)
            ],
            [
                [Hθ -> PrecipMeanField.refine_η(Hθ, k, F=F, Y=Y) for k = 1:4]...,
                [Hθ -> PrecipMeanField.refine_s²(Hθ, F=F, Y=Y) for _ = 1:4]...,
                Hθ -> PrecipMeanField.refine_aᵤ(F=F),
                Hθ -> PrecipMeanField.refine_bᵤ(Hθ, F=F),
            ]
        )

        @test precipModel.logApproxDensity([fill(2, 4)..., 2]) ≈ -2*log(2*pi) - 10;
        @test AbstractModel.getHθ(precipModel) == Hθ₀;
    end

end