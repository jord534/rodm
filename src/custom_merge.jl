include("building_tree.jl")
include("utilities.jl")
include("convert.jl")
include("struct/cluster.jl")
import Clustering
using Distances


function custom_merge(method::String = "none")
    """
    Ici, l'objectif est d'entraîner les arbres avec des clusters de données
    formés à l'intérieur des classes, suivant les règles de Kmens/Kmedioids
    les attentes sont ici uniquement sur la précison et non sur les temps de calculs
    """
    for dataSetName in ["iris", "heart", "zoo", "seeds", "wine"]
        
        print("=== Dataset ", dataSetName)
        
        # Préparation des données
        include("../data/" * dataSetName * ".txt") 
        train, test = train_test_indexes(length(Y))
        X_train = X[train,:]
        Y_train = Y[train]
        X_test = X[test,:]
        Y_test = Y[test]

        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")
        
        # Temps limite de la méthode de résolution en secondes        
        time_limit_ = 10
        faux = false
        vrai = true
        for D in 2:4
            println("\tD = ", D)
            println("\t\tUnivarié")
            testKmeans(X_train, Y_train, X_test, Y_test, D, time_limit_, faux, method)
            println("\t\tMultivarié")
            testKmeans(X_train, Y_train, X_test, Y_test, D, time_limit_, vrai, method)
        end
    end
end 

function testKmeans(X_train, Y_train, X_test, Y_test, D, time_limit::Int=-1, isMultivariate::Bool = false, method::String = "none")

    # Pour tout pourcentage de regroupement considéré
    println("\t\t\tnvl taille\t# clusters\tGap")
    for gamma in 0.2:0.2:1
        print("\t\t\t", round((1- gamma) * 100), "%\t\t")
        # On va regrouper en gamma*taille_classe chaque classe
        classes = unique(Y_train)
        #print(classes)
        clusters = Vector{Cluster}([])
        
        if method == "kmeans"
            for c in classes
                mask = findall(x -> x ==c, Y_train )
                sub_data = X_train[mask,:]
                #print(size(sub_data))
                num_of_groups = Int(round(size(sub_data)[1]*(1-gamma)))
                if num_of_groups > 0
                    sub_clusters = convert_to_cluster(kmeans(permutedims(sub_data), num_of_groups), sub_data, c)
                    clusters = vcat(clusters,sub_clusters)
                else
                    push!(clusters, Cluster(1, sub_data, Y_train[mask]) )
                end
                
            end   
        end

        if method == "kmedoids"
            for c in classes
                mask = findall(x -> x ==c, Y_train )
                sub_data = X_train[mask,:]
                #print(size(sub_data))
                
                num_of_groups = Int(round(size(sub_data)[1]*(1-gamma)))
                if num_of_groups > 0
                    distance_matrix = Matrix{Float64}(undef, size(sub_data)[1],size(sub_data)[1])
                    for i in 1:size(sub_data)[1]
                        distance_matrix[i , i] = 0
                        for j in (i+1):size(sub_data)[1]
                            dist = euclidean(sub_data[i, :], sub_data[j, :])
                            distance_matrix[i,j] = dist
                            distance_matrix[j,i] = dist
                        end
                    end

                    sub_clusters = convert_to_cluster(kmedoids(distance_matrix, num_of_groups), sub_data, c)
                    clusters = vcat(clusters,sub_clusters)
                else
                    push!(clusters, Cluster(1, sub_data, Y_train[mask]) )
                end
                
            end 
        end

        print(length(clusters), " clusters\t")
        T, obj, resolution_time, gap = build_tree(clusters, D, multivariate = isMultivariate, time_limit = time_limit)
        print(round(gap, digits = 1), "%\t") 
        print("Erreurs train/test : ", prediction_errors(T,X_train,Y_train))
        print("/", prediction_errors(T,X_test,Y_test), "\t")
        println(round(resolution_time, digits=1), "s")
    end
    println() 
end 