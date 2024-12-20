using Test

# Chemin vers le dossier contenant les tests
test_dir = "tests";

# Parcourir et exécuter chaque fichier de test
for (root, dirs, files) in walkdir(test_dir)
    for file in files
        println("Exécution de $root/$file...");
        include(joinpath(root, file));
    end
end

println("Tous les tests ont été exécutés.");