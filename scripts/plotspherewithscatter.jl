using Lebedev_Interpolations
using MoM_Basics, MoM_Kernels

## 绘图并保存
# 空球
plot_sphere_with_nodes()

for i in -1:4

    # 截断项数
    τ =  MoM_Kernels.truncationLCal(0.25*Params.λ_0* (2. ^i))
    # 多项式阶数
    p = 2τ + 1

    # 球面采样点与权重
    nodes, weights = gq_xsws_on_sphere(τ)
    plot_sphere_with_nodes(;nodes = nodes, filename = "GQ_$(τ)")

    nodes, weights = getlbSortedData(p)
    plot_sphere_with_nodes(;nodes = nodes, filename = "LQ_$(τ)")

end


# 处理生成的文件
files = filter(f -> ((startswith(f, "GQ") || startswith(f, "LQ")) && endswith(f, ".png")), readdir("results"))
clip_imag("results", files, [351:1450, 651:1750], "results\\img_cliped")