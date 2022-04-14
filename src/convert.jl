# This will try to convert a KmeansResult variable into a Vector{Cluster}
include("struct/cluster.jl")
using Clustering

"""
# here is the Cluster type architecture
mutable struct Cluster
    """
#    Représente un regroupement de données custom
    """
    dataIds::Vector{Int}
    lBounds::Vector{Float64}
    uBounds::Vector{Float64}
    x::Matrix{Float64}
    class::Int

    function Cluster()
        return new()
    end
end 

function Cluster(id::Int, x::Matrix{Float64}, y::Vector{Int})
    """
 #   Constructeur d'un cluster custom
    
 #   Entrées :
 #   - id : identifiant du premier élément du cluster
 #   - x  : caractéristique des données d'entraînement
 #   - y  : classe des données d'entraînement
    """
    c = Cluster()
    c.x = x[Vector{Int}([id]), :] # Crée une matrice contenant une ligne
    c.class = y[id]
    c.dataIds = Vector{Int}([id])
    c.lBounds = Vector{Float64}(x[id, :])
    c.uBounds = Vector{Float64}(x[id, :])

    return c
    
end 

# here the KmeansResult type architecture

struct KmeansResult{C<:AbstractMatrix{<:AbstractFloat},D<:Real,WC<:Real} <: ClusteringResult
    centers::C                 # cluster centers (d x k) d is the number of features
    assignments::Vector{Int}   # assignments (n)
    costs::Vector{D}           # cost of the assignments (n)
    counts::Vector{Int}        # number of points assigned to each cluster (k)
    wcounts::Vector{WC}        # cluster weights (k)
    totalcost::D               # total cost (i.e. objective)
    iterations::Int            # number of elapsed iterations
    converged::Bool            # whether the procedure converged
end

# Q : how is the Vector{Cluster} used ? ie, indexed ? 
# A : look in the build_tree.jl file
# # # A : not relevant, the only necessity is to convert to Vector{Cluster}




# Ultimately the goal is to write a function that converts kmeans output into a vector of clusters
using Clustering

"""
# We should define a globalized number of clusters (ex : N) that sets how many means we will have, and the size of the Vector{Cluster}
#N = 4
"""
"""

function convert_to_cluster(clustering_output::KmeansResult, data::Matrix{Float64}, class::Int)
    clusters = Vector{Cluster}([])

    num_of_groups = size(clustering_output.centers)[2]
    num_of_features = size(clustering_output.centers)[1]

    for g in 1:num_of_groups # Julia uses 1-based indexing, this is a "small" loop
        # g is the group id
        curr_cluster = Cluster()
        # let's obtain all the data points within group "g"
        idx = findall(t -> t == g, clustering_output.assignments)
        curr_cluster.dataIds = idx
        # idx is the index array of all data points in group "g"
        class_ = class

        curr_min = Inf64 
        curr_max = -Inf64
        lbound = zeros(0)
        ubound= zeros(0)
        for f in 1:num_of_features
            for d in idx
                if data[d,f] < curr_min
                    curr_min = data[d,f]
                end
                if data[d,f] > curr_max
                    curr_max = data[d,f]
                end
            end
            append!(lbound, curr_min)
            append!(ubound, curr_max)
        end
        curr_cluster.lBounds = lbound
        curr_cluster.uBounds = ubound

        curr_cluster.x = data[idx,:]
        curr_cluster.class = class_



        push!(clusters, curr_cluster)
    end

    return clusters
end 

function convert_to_cluster(clustering_output::KmedoidsResult, data::Matrix{Float64}, class::Int)
    clusters = Vector{Cluster}([])

    num_of_groups = length(clustering_output.medoids)
    num_of_features = size(data)[2]

    for g in 1:num_of_groups # Julia uses 1-based indexing, this is a "small" loop
        # g is the group id
        curr_cluster = Cluster()
        # let's obtain all the data points within group "g"
        idx = findall(t -> t == g, clustering_output.assignments)
        curr_cluster.dataIds = idx
        # idx is the index array of all data points in group "g"
        class_ = class

        curr_min = Inf64 
        curr_max = -Inf64
        lbound = zeros(0)
        ubound= zeros(0)
        for f in 1:num_of_features
            for d in idx
                if data[d,f] < curr_min
                    curr_min = data[d,f]
                end
                if data[d,f] > curr_max
                    curr_max = data[d,f]
                end
            end
            append!(lbound, curr_min)
            append!(ubound, curr_max)
        end
        curr_cluster.lBounds = lbound
        curr_cluster.uBounds = ubound

        curr_cluster.x = data[idx,:]
        curr_cluster.class = class_



        push!(clusters, curr_cluster)
    end

    return clusters
end 