using Test

include("../../src/models/PrecipMeanField.jl");
using .PrecipMeanField

@testset "PrecipMeanField.jl" begin

    Y = [float.([i*2, i*2+1]) for i = 0:3];
    F = PrecipMeanField.iGMRF(2, 2, NaN);
    Hθ = [
        [float(i) for i = 1:4],
        ones(4),
        3.0,
        2.0,
    ];

    @testset "logTargetDensity(θ; F, Y)" begin
        
        θ = [1.0, 2.0, 3.0, 4.0, exp(1)];

        @test PrecipMeanField.logTargetDensity(θ, F=F, Y=Y) ≈ 3/2 - 8 - 501/100 * exp(1);

    end


    @testset "refine_s²(Hθ; F, Y)" begin
        
        s² = PrecipMeanField.refine_s²(Hθ, F=F, Y=Y);

        @test s²[1] ≈ 2/203;
        @test s²[2] ≈ 1/103;

    end


    @testset "refine_aᵤ(m)" begin
        
        aᵤ = PrecipMeanField.refine_aᵤ(F=F);

        @test aᵤ ≈ 5.0;

    end
    
    
    @testset "refine_η(η, aᵤ, bᵤ; F, Y)" begin
        
        Y = [float.([i*2, i*2+1]) for i = 0:8];
        F = iGMRF(3, 3, NaN);
        Hθ = [
            [float(i) for i = 1:9],
            ones(9),
            3.0,
            2.0,
        ];

        @test PrecipMeanField.refine_η(Hθ, 1, F=F, Y=Y) ≈ 2.0;
        @test PrecipMeanField.refine_η(Hθ, 5, F=F, Y=Y) ≈ 94/16;

    end
    
    
    @testset "refine_bᵤ(η, F)" begin
        
        bᵤ = PrecipMeanField.refine_bᵤ(Hθ, F=F);

        @test bᵤ ≈ 5.01;

    end

end