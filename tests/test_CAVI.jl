using Test

include("../precipMeanField/CAVI.jl");

@testset "CAVI.jl" begin
    
    @testset "compute_s²(aᵤ, bᵤ, n_obs, n_neighbors)" begin
        
        aᵤ = 3.0;
        bᵤ = 2.0;
        n_obs = 100;
        n_neighbors = [1, 2];

        s² = compute_s²(aᵤ, bᵤ, n_obs, n_neighbors);

        @test s²[1] ≈ 2/203;
        @test s²[2] ≈ 1/103;

    end


    @testset "updateHyperParams!(Hθ; F, Y)" begin
        
        η = [float(i) for i = 1:9];
        bᵤ = 2;
        Hθ = [η..., bᵤ];
        F = iGMRF(3, 3, 1);
        Y = [float.([i*2, i*2+1]) for i = 0:8];

        updateHyperParams!(Hθ, F=F, Y=Y)

        @test Hθ[1] ≈ 16/7;
        @test Hθ[2] ≈ 430/133;

    end

    
    @testset "compute_aᵤ(m)" begin
        
        aᵤ = compute_aᵤ(9);

        @test aᵤ ≈ 5.0;

    end
    
    
    @testset "compute_η(η, aᵤ, bᵤ; F, Y)" begin
        
        aᵤ = 3;
        bᵤ = 2;
        F = iGMRF(3, 3, 1);
        η = [float(i) for i = 1:9];
        Y = [float.([i*2, i*2+1]) for i = 0:8];

        @test compute_η(1, η, aᵤ=aᵤ, bᵤ=bᵤ, F=F, Y=Y) ≈ 2.0;
        @test compute_η(5, η, aᵤ=aᵤ, bᵤ=bᵤ, F=F, Y=Y) ≈ 94/16;

    end
    
    
    @testset "compute_bᵤ(η, F)" begin
        
        F = iGMRF(2, 2, 1);
        η = [1, 2, 3, 4];

        bᵤ = compute_bᵤ(η, F);

        @test bᵤ ≈ 5.01;

    end

end