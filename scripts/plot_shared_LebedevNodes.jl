using Lebedev_Interpolations
using MoM_Basics, MoM_Kernels

## 绘图并保存
# 空球
plot_sphere_with_nodes()

# 截断项数
τ =  MoM_Kernels.truncationLCal(0.25*Params.λ_0* (2. ^ 0))
# 多项式阶数
p = 2τ + 1

# 球面采样点与权重
nodes, _ = getlbSortedData(p)

# 截断项数
τf =  MoM_Kernels.truncationLCal(0.25*Params.λ_0* (2. ^1))
# 多项式阶数
pf = 2τf + 1

# 球面采样点与权重
nodesf, _ = getlbSortedData(pf)
# 共享点
nodes_shared = hcat(intersect(eachcol(nodes), eachcol(nodesf))...)

# 子层父层的点画在同一张图上
fig = plot_sphere_with_nodes(;nodes = nodes, filename = "LQ_$(τ)and$(τf)")
plot_sphere_with_nodes(nodes, nodesf; colorid = [1, 3], marker = [:circle, :star4],  filename = "LQ_$(τ)and$(τf)")
plot_sphere_with_nodes!(fig; nodes = nodes_shared, colorid = 2,  filename = "LQ_$(τ)andShared")

fig = plot_sphere_with_nodes(;nodes = nodes_shared, filename = "LQ_Shared")

for nInterp in 4:4:16
    interpW = interpWeightsInitial(nodes, nodesf; nInterp=nInterp)
    fill!(interpW.nzval, 1)
    plotSparseArrayPattern(Array(interpW')[1:end, end:-1:1]; size_inches = (1, 2), filename = "LQ_$(τ)Interp$(nInterp)Pattern")#title = L"N_k = %$(nInterp)", 
end