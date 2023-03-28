
function pinv2W!(w, nInterp, xx2D, yy2D)
    # 插值点满阵则全计算
    if nInterp >= (size(w, 2) ÷ 2)
        xxpinv = pinv(xx2D)
        wFinal = yy2D * xxpinv
        w.nzval .= 1
        for j in axes(w, 2)
            for i in axes(w, 1)
                    (w[i, j] != 0) && begin
                        w[i, j] = wFinal[i, j]
                    end
                    continue
                w[i, j] = 0
            end
        end
    else# 非满阵计算
        # 进度条
        pmeter =  Progress(size(w, 1), "Calculating Interpolation weights...")

        # 对行循环计算本行插值矩阵
        @threads for irow in axes(w, 1)
            # 更新进度条
            next!(pmeter)
            # 权重行
            wrow    =   w[irow, :]
            # 跳过不用计算的极点
            nnz(wrow) == 1 && continue
            # 非零元素的列索引
            nzind   =   wrow.nzind
            # 提取 本行 右侧项
            xx      =   xx2D[nzind, :]
            # 计算伪逆
            xxpinv  =   pinv(xx)
            # 计算权重
            wi      =   view(yy2D, irow:irow, :) * xxpinv 
            # 写入结果
            w[irow, nzind]  .=  reshape(wi, :)
        end
    end

    return w

end

acc(w, x, y) = mean(abs, ( w*x .- y), dims = 1) ./ maximum(abs, y, dims = 1)

function saveInterpW2file(τt, τp, ε, w; dpath = "deps/InterpolationWeights/")

    !isdir(dpath) && mkpath(dpath)

    fileε   = try
        jldopen(joinpath(dpath, "$(τt)To$(τp).jld2"), "r") do file
            file["ε"]
        end
    catch
        1.
    end

    if fileε > ε
        @info "旧的精度: $fileε, 新的精度: ε"
        jldopen(joinpath(dpath, "$(τt)To$(τp).jld2"), "w+") do file
            @info "得到更精确结果！保存中…"
            write(file, "data", w)
            write(file, "ε", ε)
        end
        @info "已保存结果。"
    else
        @info "比上次结果差，不保存"
    end

    nothing
    
end

function calWFinal(cubeL; nInterp, xx2D, yy2D, xx2Dte, yy2Dte, FT = Precision.FT)
    rel_l::FT = cubeL/Params.λ_0
    calWFinal(;rel_l = rel_l, nInterp = nInterp, xx2D = xx2D, yy2D = yy2D, xx2Dte=xx2Dte, yy2Dte=yy2Dte, FT = FT)
end

"""
采用伪逆计算插值矩阵，对稀疏矩阵要分行计算
f(k̂ₗ₋₁) = Wₗ₋₁ₗ f(k̂ₗ)
Wₗ₋₁ₗ = pinv(f(k̂ₗ)) f(k̂ₗ₋₁) 
"""
function calWFinal(;rel_l, nInterp, xx2D, yy2D, xx2Dte, yy2Dte, FT = Precision.FT)
    # truncL
    τt  =   truncationLCal(rel_l = rel_l)
    τp  =   truncationLCal(rel_l = 2rel_l)
    # poles
    tnodes = get_t_nodes(τt)
    pnodes = get_t_nodes(τp)

    # 初始化权重
    w = interpWeightsInitial(tnodes, pnodes; nInterp = nInterp)

    # 计算权重
    pinv2W!(w, nInterp, xx2D, yy2D)

    # 计算误差
    ε = acc(w, xx2Dte, yy2Dte)

    @info "测试误差" nInterp = nInterp, ε = ε

    saveInterpW2file(τt, τp, ε, w)

    return nothing
end

function runpinvCal(cubeL; FT = Precision.FT)
    rel_l::FT = cubeL/Params.λ_0
    runpinvCal(; rel_l = rel_l, FT = FT)
end

function runpinvCal(; rel_l, nInterp = 9, FT = Precision.FT)

    # 生成数据集
    tArray, pArray = generate_dataset_on_cubeEdgeL(rel_l = rel_l)

    # 划分数据集
    flag = trunc(Int, 0.8*size(tArray, 2))
    @views xx2D = tArray[:, 1:flag]
    @views yy2D = pArray[:, 1:flag]
    @views xx2Dte = tArray[:, (flag+1):end]
    @views yy2Dte = pArray[:, (flag+1):end]

    # 权重 计算
    calWFinal(;rel_l = rel_l, nInterp = nInterp, xx2D = xx2D, yy2D = yy2D, xx2Dte=xx2Dte, yy2Dte=yy2Dte, FT = FT)

    nothing

end
