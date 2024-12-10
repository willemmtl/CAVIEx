using Test, Distributions

include("../precipFramework/iGMRF.jl");
include("../precipFramework/utils.jl");
include("../precipMeanField/model.jl");

@testset "framework.jl" begin
    
    @testset "logFunctionalFormPosterior(θ; F, Y)" begin
        
        Y = [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0]
        ];
        F = iGMRF(2, 2, 1);
        θ = [1.0, 2.0, 3.0, 4.0, exp(1)];

        @test logFunctionalFormPosterior(θ, F=F, Y=Y) ≈ 3/2 - 8 - 501/100 * exp(1)

    end


    @testset "evaluateLogMvDensity(f, supp)" begin

        supp = [1 2; 3 4];
        
        f(x::DenseVector) = x[1] + x[2]^2;

        res = evaluateLogMvDensity(f, supp);

        @test res[1] == 10;
        @test res[2] == 18;

    end
    
    
    @testset "findMode(f, θ₀)" begin
        
        f(θ::DenseVector) = pdf(MvNormal(zeros(2), I), θ);
        θ₀ = [1.0, 1.0];
        mode = findMode(f, θ₀);
        
        @test isapprox(Optim.minimizer(mode), [0, 0], atol=1e-15)
        @test -Optim.minimum(mode) ≈ 1 / (2*pi)

    end


    @testset "computeFisherInformation(logf, θ̂)" begin
        
        μ = 1.0;
        σ² = 2.5;

        x̂ = μ; # mode

        logf(θ::DenseVector) = -0.5 * log(θ[2]) - (x̂ - θ[1])^2 / (2*θ[2]);

        @test computeFisherInformation(logf, [μ, σ²]) ≈ [1/σ² 0; 0 -1/(2*σ²^2)]
    
    end


    @testset "computeFisherVariance(logf, θ̂)" begin
        
        μ = 1.0;
        σ² = 2.5;

        x̂ = μ; # mode

        logf(θ::DenseVector) = -0.5 * log(θ[2]) - (x̂ - θ[1])^2 / (2*θ[2]);

        @test computeFisherVariance(logf, [μ, σ²]) ≈ [σ² 0; 0 -2*σ²^2]
    
    end

end