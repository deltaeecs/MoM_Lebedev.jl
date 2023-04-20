using MoM_Kernels:get_leafCubeSize
ws = [-4/5 9/20 9/20 9/20 9/20]

# """
#     generate_dataset_on_poles(r̂sθsϕs, tArray; ρhat = random_rhat(), rbmrp = √3/2*get_leafCubeSize()*random_rhat(), FT = Precision.FT)
# TBW
# """
# function generate_dataset_on_poles(r̂sθsϕs, tArray; ρhat = random_rhat(), rbmrp = √3/2*get_leafCubeSize()*random_rhat(), FT = Precision.FT)
#     # 常数
#     JK_0 = Params.JK_0

#     for iPole in eachindex(r̂sθsϕs)
#         # 该多极子
#         poler̂θϕ =   r̂sθsϕs[iPole]
#         # 公用的 指数项
#         exptemp =   exp(-JK_0*(poler̂θϕ.r̂ ⋅ rbmrp))
#         # 将结果写入目标数组
#         tArray[iPole, 1] = (poler̂θϕ.θhat ⋅ ρhat) * exptemp
#         tArray[iPole, 2] = (poler̂θϕ.ϕhat ⋅ ρhat) * exptemp
#     end # iPole

#     return tArray

# end

"""
    generate_dataset_on_poles(r̂sθsϕs, tArray; rvec = random_rvec(), FT = Precision.FT)


    simulate agg on basis functions.
TBW
"""
function generate_dataset_on_poles(r̂sθsϕs, tArray; rvec = random_rvec(), FT = Precision.FT)
    # 常数
    JK_0 = Params.JK_0
    # rvecp rmvec
    rvecp = rvec
    rvecm = rvec .+ random_rhat().*(rand(truncated(Normal(0.08Params.λ_0, 0.04Params.λ_0), 0, 0.12Params.λ_0)))
    offsets = [[random_rhat()...].*(rand(truncated(Normal(0.02Params.λ_0, 0.01Params.λ_0), 0.01Params.λ_0, 0.03Params.λ_0))) for _ in eachindex(ws)]
    offsets[1] .= 0

    r0p = rvecp .+ rvecp .- rvecm .+ random_rhat() .* 0.005Params.λ_0
    r0m = rvecm .+ rvecm .- rvecp .+ random_rhat() .* 0.005Params.λ_0

    rp  = copy(r0p)
    rm  = copy(r0m)

    ρhatp_iw = copy(r0p)
    ρhatm_iw = copy(r0m)
    
    for iPole in eachindex(r̂sθsϕs)
        # 该多极子
        poler̂θϕ =   r̂sθsϕs[iPole]
        for iw in eachindex(ws)

            rp .= rvecp .+ offsets[iw]
            rm .= rvecm .+ offsets[iw]

            ρhatp_iw .=  rp .- r0p
            ρhatm_iw .=  rm .- r0m
            
            # 公用的 指数项
            wpexptemp =   ws[iw]*exp(JK_0*(poler̂θϕ.r̂ ⋅ rp))
            wmexptemp =   ws[iw]*exp(JK_0*(poler̂θϕ.r̂ ⋅ rm))
            # 将结果写入目标数组
            tArray[iPole, 1] += (poler̂θϕ.θhat ⋅ ρhatp_iw) * wpexptemp
            tArray[iPole, 1] -= (poler̂θϕ.θhat ⋅ ρhatm_iw) * wmexptemp
            tArray[iPole, 2] += (poler̂θϕ.ϕhat ⋅ ρhatp_iw) * wpexptemp
            tArray[iPole, 2] -= (poler̂θϕ.ϕhat ⋅ ρhatm_iw) * wmexptemp
        end
    end # iPole

    return tArray

end

# """
#     generate_dataset_on_poles(r̂sθsϕs; ρhat = random_rhat(), rbmrp = √3/2*get_leafCubeSize()*random_rhat(), FT = Precision.FT)
    
# TBW
# """
# function generate_dataset_on_poles(r̂sθsϕs; ρhat = random_rhat(), rbmrp = √3/2*get_leafCubeSize()*random_rhat(), FT = Precision.FT)

#     # 目标数组
#     tArray = zeros(Complex{FT}, length(r̂sθsϕs), 2)
#     generate_dataset_on_poles(r̂sθsϕs, reshape(tArray, length(r̂sθsϕs), 2); ρhat = ρhat, rbmrp = rbmrp, FT = FT)
#     return tArray

# end

"""
    generate_dataset_on_poles(r̂sθsϕs; ρhat = random_rhat(), rbmrp = √3/2*get_leafCubeSize()*random_rhat(), FT = Precision.FT)
    
TBW
"""
function generate_dataset_on_poles(r̂sθsϕs; rvec = random_rvec(), FT = Precision.FT)

    # 目标数组
    tArray = zeros(Complex{FT}, length(r̂sθsϕs), 2)
    generate_dataset_on_poles(r̂sθsϕs, reshape(tArray, length(r̂sθsϕs), 2); rvec = rvecFT = FT)
    return tArray

end


"""
在球面生成随机向量
"""
function random_rvec(bd=1; FT = Precision.FT)
    [(2rand()-1)*bd, (2rand()-1)*bd, (2rand()-1)*bd]
end

"""
    generate_dataset_on_trunct(cubeEdgeL; FT = Precision.FT)

TBW
"""
function generate_dataset_on_pkpt(pk::T, pt::T, rel_l = find_zero(x -> truncation_kernel(x) - (pk+1)÷2, 0); FT = Precision.FT) where{T<:Integer}
    # trunc
    τt  =   (pk - 1) ÷ 2
    τp  =   (pt - 1) ÷ 2

    # 多项式阶数
    pt = 2τt+1
    # 若本层已超出Lebedev求积点取值范围则报错
    pt > maximum(keys(p2nDict)) && throw("多项式阶数已超出Lebedev求积点取值范围。")

    # 生成基函数矢量
    # ρhats, _ = getlbSortedData(13)
    ρhats = zeros(FT, 3, 50)
    for i in axes(ρhats, 2)
        ρhats[:, i] = random_rhat()
    end
    # 空间位置矢量
    # rbmrps = getlbSortedData(13)[1] .* (√3/2*rel_l*Params.λ_0)
    rbmrps = zeros(FT, 3, 500)
    @info "box size" (rel_l*Params.λ_0/2)
    for i in axes(rbmrps, 2)
        rbmrps[:, i]   .= random_rvec() .* (rel_l*Params.λ_0/2)
    end

    # nodes
    tnodes = get_t_nodes(τt)
    pnodes = get_t_nodes(τp)
    # r̂sθsϕs
    tr̂sθsϕs = nodes2Poles(tnodes)
    pr̂sθsϕs = nodes2Poles(pnodes)

    # 预分配内存
    tArray  = zeros(Complex{FT}, length(tr̂sθsϕs), 2, size(ρhats, 2), size(rbmrps, 2))
    pArray  = zeros(Complex{FT}, length(pr̂sθsϕs), 2, size(ρhats, 2), size(rbmrps, 2))

    # 开始计算
    pmeter =  Progress(size(rbmrps, 2), "计算数据集中…")
    for ir in axes(rbmrps, 2)#@threads 
        for iρ in axes(ρhats, 2)
            # @views generate_dataset_on_poles(tr̂sθsϕs, tArray[:, :, iρ, ir]; ρhat = ρhats[:, iρ], rbmrp = rbmrps[:, ir])
            # @views generate_dataset_on_poles(pr̂sθsϕs, pArray[:, :, iρ, ir]; ρhat = ρhats[:, iρ], rbmrp = rbmrps[:, ir])
            @views generate_dataset_on_poles(tr̂sθsϕs, tArray[:, :, iρ, ir]; rvec = rbmrps[:, ir])
            @views generate_dataset_on_poles(pr̂sθsϕs, pArray[:, :, iρ, ir]; rvec = rbmrps[:, ir])
        end
        next!(pmeter)
    end

    return reshape(tArray, length(tr̂sθsϕs)*2, :), reshape(pArray, length(pr̂sθsϕs)*2, :)

end