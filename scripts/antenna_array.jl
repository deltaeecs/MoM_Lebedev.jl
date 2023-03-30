
"""
和阵列
"""
function dipoleArray(deg::FT; arraysize = 0.46, nxny = nothing) where{FT}

    nx, ny = if isnothing(nxny)
        temp    =   floor(Int, arraysize ÷ Params.λ_0*2)
        n1Dx    =   isodd(temp) ? temp-1 : temp
        # temp    =   floor(Int, arraysize ÷ Params.λ_0*2)
        n1Dy    =   isodd(temp) ? temp-1 : temp
        n1Dx, n1Dy
    else
        nxny
    end

    antennaArray((nx, ny), (0., -deg*π/180, 0.); sourceConstructer = MagneticDipole,  
                    sourceT = MagneticDipole, soruceorientlc = (π/2, π/2, 0.), 
                    orientunit = :rad, coefftype = :taylor)
end


"""
差阵列
"""
function dipolediffArray(deg::FT; arraysize = 0.46, nxny = nothing) where{FT}
    ary     =   dipoleArray(deg::FT; arraysize = arraysize, nxny = nxny)
    setdiffArray!(ary, 2)
    ary
end
