using Lebedev_Interpolations
using MoM_Basics, MoM_Kernels
using LaTeXStrings
using ElectronDisplay

# 截断项数
τ =  MoM_Kernels.truncationLCal(0.25*Params.λ_0)

# 球面采样点与权重
nodes, weights = gq_xsws_on_sphere(τ)

# 转换到球坐标
rθϕnodes = reduce(hcat, map(cart2sphere, eachcol(nodes)))

# 绘图并保存
# fig = viz_data_in_thetaphi_plane(rθϕnodes[[3,2], :], title = L"\tau = %$(τ)", filename = "θϕnodes1", size_inches = (3.2, 2.4), fontsize = 10)


# 父层
τf =  MoM_Kernels.truncationLCal(0.25*Params.λ_0*2)
# 球面采样点与权重
nodesf, _ = gq_xsws_on_sphere(τf)
# 转换到球坐标
rθϕnodesf = reduce(hcat, map(cart2sphere, eachcol(nodesf)))

fig = viz_data_in_thetaphi_plane(rθϕnodes[[3,2], :], rθϕnodesf[[3,2], :], title = L"(\tau = %$(τ)) to (\tau = %$(τf))", filename = "θϕnodesk2f", size_inches = (3.2, 2.4), fontsize = 10)


# Lebedev
# 截断项数
τ =  MoM_Kernels.truncationLCal(0.25*Params.λ_0*8)
nodes, weights = getlbSortedData(2τ+1)
rθϕnodes = reduce(hcat, map(cart2sphere, eachcol(nodes)))
fig = viz_data_in_thetaphi_plane(rθϕnodes[[3,2], :], title = L"\tau = %$(τ)", filename = "θϕnodesLebedev32", size_inches = (3.2, 2.4), fontsize = 10)
