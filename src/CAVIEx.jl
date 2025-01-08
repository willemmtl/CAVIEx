module CAVIEx

export 
    runCAVI,
    MonteCarloKL,
    refineHyperParams!,
    initialize!,
    syncMarginals!,
    getHpValues,
    generateApproxSample,
    runMCMC!,
    evaluateLogMvDensity,
    adaptHp!,
    plotConvergenceCriterion,
    plotTrace,
    plotApproxVSMCMC

include("CAVI.jl");
include("plotting.jl");

end