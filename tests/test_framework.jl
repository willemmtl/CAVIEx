using Test, Distributions

include("../iGMRF.jl");
include("../framework.jl");

@testset "framework.jl" begin
    
    @testset "logFunctionalFormPosterior(θ; F, Y)" begin
        
        Y = [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0]
        ];
        F = iGMRF(2, 2, 1);
        θ = [exp(1), 1.0, 2.0, 3.0, 4.0];

        @test logFunctionalFormPosterior(θ, F=F, Y=Y) ≈ 3/2 - 8 - 501/100 * exp(1)

    end
    
    
    @testset "findMode(f, θ₀)" begin
        
        f(θ::DenseVector) = pdf(MvNormal(zeros(2), I), θ);
        θ₀ = [1.0, 1.0];
        mode = findMode(f, θ₀);
        
        @test isapprox(Optim.minimizer(mode), [0, 0], atol=1e-15)
        @test -Optim.minimum(mode) ≈ 1 / (2*pi)

    end


    @testset "computeFisherInformation(f, θ̂)" begin
        
        μ = 1.0;
        σ² = 2.5;

        x̂ = μ; # mode

        logf(θ::DenseVector) = -0.5 * log(θ[2]) - (x̂ - θ[1])^2 / (2*θ[2]);

        @test computeFisherInformation(logf, [μ, σ²]) ≈ [1/σ² 0; 0 -1/(2*σ²^2)]
    
    end


    @testset "computeFisherVariance(f, θ̂)" begin
        
        μ = 1.0;
        σ² = 2.5;

        x̂ = μ; # mode

        logf(θ::DenseVector) = -0.5 * log(θ[2]) - (x̂ - θ[1])^2 / (2*θ[2]);

        @test computeFisherVariance(logf, [μ, σ²]) ≈ [σ² 0; 0 -2*σ²^2]
    
    end

end