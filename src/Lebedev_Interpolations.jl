module Lebedev_Interpolations

using NearestNeighbors, JLD2
using SparseArrays, Statistics
# using Plots, ElectronDisplay
using GLMakie, CairoMakie
using Colors, ColorSchemes
using LaTeXStrings
using ElectronDisplay
using Images, FileIO, ProgressMeter
using MoM_Basics, MoM_Kernels
using LinearAlgebra
using .Threads

export  getlbSortedData,
        plot_sphere_with_nodes,
        plot_sphere_with_nodes!,
        viz_data_in_thetaphi_plane,
        clip_imag,
        interpWeightsInitial,
        plotSparseArrayPattern,
        common_faces,
        get_t_nodes,
        generate_dataset_on_poles,
        generate_dataset_on_cubeEdgeL,
        runpinvCal

# Lebedev数据集
include("LebedevSortedPoints.jl")
# Lebedev矢量插值
include("LVI.jl")

# 数据集生成函数
include("dataset_generator.jl")

# 插值权重训练函数
include("pinv2interpW.jl")

# GLMakie 绘图
include("visualizing.jl")

# 工具函数
include("utlis.jl")

end # module Lebedev_Interpolations
