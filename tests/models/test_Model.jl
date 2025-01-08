using Test, Distributions, GMRF, OrderedCollections

include("ressources/model.jl");
include("../../src/instances/InstanceDemo.jl");

@testset "Model.jl" begin

    @testset "Model" begin

        @test modelBase.params[:μ].numero == 1;
        @test modelBase.params[:μ].approxDistribution == Normal;
        @test modelBase.nParams == 2;
        @test modelBase.hyperParams[:a].numero == 3;
        @test modelBase.hyperParams[:η].refiningFunction(0) == 1;
        @test modelBase.nHyperParams == 4;
        @test modelBase.pHp[:μ] == Dict(:μ => :η, :σ => :s);

    end


    @testset "isolateHyperParams(parameters)" begin
        
        modelParse = CAVIEx.Model(
            paramsToParse,
            () -> nothing,
            () -> nothing,
            () -> nothing,
        );

        @test modelParse.hyperParams[:s²3].numero == 12;

    end


    @testset "syncMarginals!(model)" begin
        
        syncMarginals!(modelSync);
        
        @test modelSync.params[:μ].approxMarginal == Normal(0.0, 2.0);
        @test modelSync.params[:σ].approxMarginal == Gamma(1.0, .5);
        
        # Turn one of the field name for Gamma into a wrong one
        # β instead of θ
        modelSync.pHp[:σ][:β] = :b;
        delete!(modelSync.pHp[:σ], :θ)

        @test_throws ErrorException syncMarginals!(modelSync);

    end


    @testset "getHpValues(model)" begin
        
        @test getHpValues(modelSync) == [0.0, 4.0, 1.0, 2.0];

    end


    @testset "initialize!(model, hp0)" begin
        
        hp0 = [0.0, 1.0, 1.0, 1.0];

        initialize!(modelBase, hp0)

        @test modelBase.hyperParams[:s].trace[end] == 1.0;

    end


    @testset "generateApproxSample(model, N)" begin

        N = 100;
        supp = generateApproxSample(modelBase, N);

        @test size(supp, 1) == 2;

    end


    @testset "runMCMC!(model, niter)" begin
        
        instance = InstanceDemo(
            seed=400,
            realmu=75.0,
            realsigma2=100.0,
        );

        @test isnothing(instance.model.params[:μ].mcmcSample);

        
        niter = 10;
        runMCMC!(instance.model, niter);
        
        @test length(instance.model.params[:μ].mcmcSample) == niter;

    end

end