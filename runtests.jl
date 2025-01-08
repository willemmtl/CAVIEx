using Test
using Revise

if !isdefined(Main, :CAVIEx)
    include("src/CAVIEx.jl");
end
using .CAVIEx

# Chemin vers le dossier contenant les tests
test_dir = "tests";

# Parcourir et exécuter chaque fichier de test
for (root, dirs, files) in walkdir(test_dir)
    for file in files
        if file[1:4] == "test"
            println("Exécution de $root/$file...");
            include(joinpath(root, file));
        end
    end
end

println("Tous les tests ont été exécutés.");