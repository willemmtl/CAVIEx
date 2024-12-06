using Test, Random, Distributions

include("../precipModel/iGMRF.jl");

@testset "iGMRF.jl" begin

    @testset "buildStructureMatrix(m₁, m₂)" begin
        
        W = buildStructureMatrix(2, 2);

        @test W[1, 1] == 2;
        @test W[1, 2] == -1;
        @test W[2, 3] == 0;

    end

    @testset "sampleIGMRF(F)" begin
        
        F = iGMRF(2, 2, 1);
        
        m = F.G.m₁ * F.G.m₂;
        e₁ = ones(m, 1);
        Q = F.G.W + e₁ * e₁';
        Σ = inv(Q) - inv(Q) * e₁ * inv(e₁'*inv(Q)*e₁) * e₁' * inv(Q);

        # Generate samples
        n = 100000;
        samples = zeros(m, n);
        for i =1:n
            samples[:, i] = sampleIGMRF(F)
        end

        # Empirical mean
        x̄ = sum(samples, dims=2) / n;
        # Empirical variance
        Δx = samples .- x̄;
        Σ̄ = (Δx * Δx') / (n-1);

        @test norm(Σ̄.-Σ) < .01;
        @test abs((e₁' * samples[:, div(n, 2)])[1])  < 1e-7;

    end

end