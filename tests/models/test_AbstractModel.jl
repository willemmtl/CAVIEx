using Test, GMRF

include("../../src/models/AbstractModel.jl");
using .AbstractModel
include("../../src/models/PrecipMeanField.jl");
using .PrecipMeanField

@testset "AbstractModel.jl" begin

    F = iGMRF(2, 2, 1, NaN);
    Y = [
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
        [6.0, 7.0],
    ];
    hyperParams = [["η$k" for k = 1:4]..., ["s²$k" for k = 1:4]..., "aᵤ", "bᵤ"];

    precipModel = AbstractModel.BaseModel(
        hyperParams,
        θ -> PrecipMeanField.logTargetDensity(θ, F=F, Y=Y),
        (θ, Hθ) -> PrecipMeanField.logApproxDensity(θ, Hθ, F=F),
        [
            [Hθ -> PrecipMeanField.μMarginal(Hθ, k=k, F=F) for k = 1:4]...,
            Hθ -> PrecipMeanField.κMarginal(Hθ, F=F)
        ],
        [
            [Hθ -> PrecipMeanField.refine_η(Hθ, k, F=F, Y=Y) for k = 1:4]...,
            [Hθ -> PrecipMeanField.refine_s²(Hθ, k, F=F, Y=Y) for k = 1:4]...,
            Hθ -> PrecipMeanField.refine_aᵤ(F=F),
            Hθ -> PrecipMeanField.refine_bᵤ(Hθ, F=F),
        ]
    )
    Hθ₀ = [zeros(4)..., ones(4)..., 1.0, 1.0];


    @testset "BaseModel(hyperParams, logTargetDensity, logApproxMarginals, refiningFunctions)" begin
        
        @test isnothing(precipModel.hyperParamsTrace["η1"]);
        @test isnothing(precipModel.hyperParamsValue[1]);
        @test precipModel.logApproxDensity([fill(2, 4)..., 2], Hθ₀) ≈ -2*log(2*pi) - 10;
    
    end


    @testset "storeValues(model)" begin
        
        for (k, value) in enumerate(Hθ₀)
            precipModel.hyperParamsValue[k] = value;
        end
        AbstractModel.storeValues(precipModel);
        @test precipModel.hyperParamsTrace["s²1"][end] == 1.0;
        
        precipModel.hyperParamsValue[5] = nothing;
        @test_throws ErrorException AbstractModel.storeValues(precipModel);

    end

end