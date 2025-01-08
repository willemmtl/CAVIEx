using Test, GMRF

if !isdefined(Main, :PrecipMeanField)
    include("../../src/models/PrecipMeanField.jl");
end
using .PrecipMeanField

@testset "PrecipMeanField.jl" begin

    Y = [float.([i*2, i*2+1]) for i = 0:3];
    F = iGMRF(2, 2, 1, NaN);
    Hθ = [
        [float(i) for i = 1:4]...,
        ones(4)...,
        3.0,
        2.0,
    ];

    @testset "refine_s²(Hθ, k; F, Y)" begin

        s² = PrecipMeanField.refine_s²(Hθ, 1, F=F, Y=Y);

        @test s² ≈ 0.2;

    end


    @testset "refine_aᵤ(m)" begin
        
        aᵤ = PrecipMeanField.refine_aᵤ(F=F);

        @test aᵤ ≈ 2.5;

    end
    

    @testset "refine_bᵤ(η, F)" begin
        
        bᵤ = PrecipMeanField.refine_bᵤ(Hθ, F=F);

        @test bᵤ ≈ 5.01;

    end


    @testset "logTargetDensity(θ; F, Y)" begin
        
        Y = [float.([i, i]) for i = 1:4]
        θ = [1.0, 2.0, 3.0, 4.0, exp(1)];

        @test PrecipMeanField.logTargetDensity(θ, F=F, Y=Y) ≈ 3/2 - 8 - 501/100 * exp(1);

    end

    
    @testset "refine_η(η, aᵤ, bᵤ; F, Y)" begin
        
        m₁ = 3;
        m₂ = 3;
        m = m₁ * m₂;
        Y = [float.([i*2, i*2+1]) for i = 0:m];
        F = iGMRF(m₁, m₂, 1, NaN);
        Hθ = [
            [float(i) for i = 1:9]...,
            ones(9)...,
            3.0,
            2.0,
        ];

        @test PrecipMeanField.refine_η(Hθ, 1, F=F, Y=Y) ≈ 2.0;
        @test PrecipMeanField.refine_η(Hθ, 5, F=F, Y=Y) ≈ 94/16;

    end
    
end