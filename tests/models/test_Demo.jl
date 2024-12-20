using Test, GMRF

if !isdefined(Main, :Demo)
    include("../../src/models/Demo.jl");
end
using .Demo

@testset "Demo.jl" begin

    Y = [float(i) for i = 1:4];
    Hθ = [1.0, 1.0, 0.0, 1.0];

    @testset "logTargetDensity(θ; Y)" begin
        
        θ = [0.0, 1.0];

        @test Demo.logTargetDensity(θ, Y=Y) ≈ -15.0;

    end


    @testset "refine_α(; Y)" begin

        α = Demo.refine_α(Y=Y);

        @test α ≈ 2.0;

    end


    @testset "refine_β(Hθ; Y)" begin
        
        β = Demo.refine_β(Hθ, Y=Y);

        @test β ≈ 17.0;

    end
    

    @testset "refine_m(; Y)" begin
        
        m = Demo.refine_m(Y=Y);

        @test m ≈ 2.5;

    end


    @testset "refine_s²(Hθ; Y)" begin
        
        @test Demo.refine_s²(Hθ, Y=Y) ≈ 1/3;

    end

end