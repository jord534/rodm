include("building_tree.jl")
include("utilities.jl")
include("convert.jl")
import Clustering
import Statistics
using Distances
import StatsBase

function custom_main(N::Int = -1, method::String= "none")
    ############################################################################################
    # N correspond à la réduction du nombre de datapoint en pourcentage
    # exemple : si N = 30, alors on réduit le nombre de datapoint de 30% et on transforme 
    # 1000 points de données en 700 points obtenus par method

    # method est soit "kmeans", "kmedoids", ou "none", sinon renvoi une erreur
    ############################################################################################

    """
    Ici, l'objectif est d'entraîner les arbres avec les centres des groupes Kmeans/Kmedioids plutôt
    qu'avec le jeu de données en entier

    Les tests se feront avec les données complètes

    on espère réduire les temps de calculs principalement
    ensuite, on peut s'attendre à une précision optimale atteinte pour un certain nombre de groupes par classe 

    """
    # Pour chaque jeu de données
    for dataSetName in ["iris", "heart", "zoo", "seeds", "wine"]
        
        print("=== Dataset ", dataSetName)
        

        # Préparation des données
        include("../data/" * dataSetName * ".txt") 
        train, test = train_test_indexes(length(Y))
        X_train = X[train, :]
        Y_train = Y[train]
        # ici nous allons appliquer kmeans à X_train
        # la classe associée à chaque groupe sera la classe majoritaire du groupe
        if method == "kmeans"
            num_of_groups = Int(round(size(X_train)[1]*(1-N/100))) # computing how many groups we'd need to shrink by N percent
            groups = kmeans(permutedims(X_train), num_of_groups)
            X_train_prime = permutedims(groups.centers)
            Y_train_prime = Vector{Int}([])
            for g in 1:num_of_groups 
                mask = findall(y -> y == g, groups.assignments)
                class_of_g = StatsBase.mode(Y_train[mask])
                push!(Y_train_prime, class_of_g)
            end
        end
        if method == "kmedoids"
            num_of_groups = Int(round(size(X_train)[1]*(1-N/100))) # computing how many groups we'd need to shrink by N percent
            distance_matrix = Matrix{Float64}(undef, size(X_train)[1],size(X_train)[1])
            for i in 1:size(X_train)[1]
                distance_matrix[i , i] = 0
                for j in (i+1):size(X_train)[1]
                    dist = euclidean(X_train[i, :], X_train[j, :])
                    distance_matrix[i,j] = dist
                    distance_matrix[j,i] = dist
                end
            end
            groups = kmedoids(distance_matrix, num_of_groups)
            X_train_prime = X_train[groups.medoids,:]
            Y_train_prime = Vector{Int}([])
            for g in 1:num_of_groups 
                mask = findall(y -> y == g, groups.assignments)
                class_of_g = StatsBase.mode(Y_train[mask])
                push!(Y_train_prime, class_of_g)
            end
        end
        X_test = X[test, :]
        Y_test = Y[test]
        if method == "none"
            println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")
            X_train_prime = X_train
            Y_train_prime = Y_train
        else
            println(" (train size after shrinkage ", size(X_train_prime, 1), ", test size is unchanged ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")
        end
        # Temps limite de la méthode de résolution en secondes
        time_limit = 30

        # Pour chaque profondeur considérée
        for D in 2:4

            println("  D = ", D)

            ## 1 - Univarié (séparation sur une seule variable à la fois)
            # Création de l'arbre
            print("    Univarié...  \t")
            #T, obj, resolution_time, gap = build_tree(X_train, Y_train, D,  multivariate = false, time_limit = time_limit)
            T, obj, resolution_time, gap = build_tree(X_train_prime, Y_train_prime, D,  multivariate = false, time_limit = time_limit)
            # Test de la performance de l'arbre
            print(round(resolution_time, digits = 1), "s\t")
            print("gap ", round(gap, digits = 1), "%\t")
            if T != nothing
                print("Erreurs train/test ", prediction_errors(T,X_train,Y_train))
                print("/", prediction_errors(T,X_test,Y_test), "\t")
            end
            println()

            ## 2 - Multivarié
            print("    Multivarié...\t")
            #T, obj, resolution_time, gap = build_tree(X_train, Y_train, D, multivariate = true, time_limit = time_limit)
            T, obj, resolution_time, gap = build_tree(X_train_prime, Y_train_prime, D, multivariate = true, time_limit = time_limit)
            print(round(resolution_time, digits = 1), "s\t")
            print("gap ", round(gap, digits = 1), "%\t")
            if T != nothing
                print("Erreurs train/test ", prediction_errors(T,X_train,Y_train))
                print("/", prediction_errors(T,X_test,Y_test), "\t")
            end
            println("\n")
        end
    end 
end
