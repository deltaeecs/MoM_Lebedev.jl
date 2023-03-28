using MoM_Kernels:get_leafCubeSize

"""
    generate_dataset_on_poles(r̂sθsϕs, tArray; ρhat = random_rhat(), rbmrp = √3/2*get_leafCubeSize()*random_rhat(), FT = Precision.FT)
TBW
"""
function generate_dataset_on_poles(r̂sθsϕs, tArray; ρhat = random_rhat(), rbmrp = √3/2*get_leafCubeSize()*random_rhat(), FT = Precision.FT)
    # 常数
    JK_0 = Params.JK_0

    for iPole in eachindex(r̂sθsϕs)
        # 该多极子
        poler̂θϕ =   r̂sθsϕs[iPole]
        # 公用的 指数项
        exptemp =   exp(-JK_0*(poler̂θϕ.r̂ ⋅ rbmrp))
        # 将结果写入目标数组
        temp = (poler̂θϕ.θhat ⋅ ρhat) * exptemp
        tArray[iPole, 1, 1] =  real(temp)
        tArray[iPole, 1, 2] =  imag(temp)
        temp =  (poler̂θϕ.ϕhat ⋅ ρhat) * exptemp
        tArray[iPole, 2, 1] =  real(temp)
        tArray[iPole, 2, 2] =  imag(temp)
    end # iPole

    return tArray

end

"""
    generate_dataset_on_poles(r̂sθsϕs; ρhat = random_rhat(), rbmrp = √3/2*get_leafCubeSize()*random_rhat(), FT = Precision.FT)
    
TBW
"""
function generate_dataset_on_poles(r̂sθsϕs; ρhat = random_rhat(), rbmrp = √3/2*get_leafCubeSize()*random_rhat(), FT = Precision.FT)

    # 目标数组
    tArray = zeros(FT, length(r̂sθsϕs), 4)
    generate_dataset_on_poles(r̂sθsϕs, reshape(tArray, length(r̂sθsϕs), 2, 2); ρhat = ρhat, rbmrp = rbmrp, FT = FT)
    return tArray

end


"""
    generate_dataset_on_trunct(cubeEdgeL; FT = Precision.FT)

TBW
"""
function generate_dataset_on_cubeEdgeL(cubeL; FT = Precision.FT)

    rel_l::FT = cubeL/Params.λ_0

    return generate_dataset_on_cubeEdgeL(; rel_l=rel_l, FT=FT)

end
"""
    generate_dataset_on_trunct(cubeEdgeL; FT = Precision.FT)

TBW
"""
function generate_dataset_on_cubeEdgeL(;rel_l, FT = Precision.FT)

    τt  =   truncationLCal(rel_l = rel_l)
    τp  =   truncationLCal(rel_l = 2rel_l)
    # 多项式阶数
    pt = 2τt+1
    # 若本层已超出Lebedev求积点取值范围则报错
    pt > maximum(keys(p2nDict)) && throw("多项式阶数已超出Lebedev求积点取值范围。")

    # 生成基函数矢量
    ρhats, _ = getlbSortedData(13)
    # 空间位置矢量
    rbmrps = getlbSortedData(13)[1] .* (√3/2*rel_l*Params.λ_0)

    # nodes
    tnodes = get_t_nodes(τt)
    pnodes = get_t_nodes(τp)
    # r̂sθsϕs
    tr̂sθsϕs = nodes2Poles(tnodes)
    pr̂sθsϕs = nodes2Poles(pnodes)

    # 预分配内存
    tArray  = zeros(FT, length(tr̂sθsϕs), 2, 2, size(ρhats, 2), size(rbmrps, 2))
    pArray  = zeros(FT, length(pr̂sθsϕs), 2, 2, size(ρhats, 2), size(rbmrps, 2))
    for ir in axes(rbmrps, 2),  iρ in axes(ρhats, 2)
        @views generate_dataset_on_poles(tr̂sθsϕs, tArray[:, :, :, iρ, ir]; ρhat = ρhats[:, iρ], rbmrp = rbmrps[:, ir])
        @views generate_dataset_on_poles(pr̂sθsϕs, pArray[:, :, :, iρ, ir]; ρhat = ρhats[:, iρ], rbmrp = rbmrps[:, ir])
    end

    return reshape(tArray, length(tr̂sθsϕs)*2, :), reshape(pArray, length(pr̂sθsϕs)*2, :)

end