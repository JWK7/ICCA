#Calculating L M and p for Extrinsic CCA
function findLMp(df1,df2)
	Rxx = Cov2Sets(df1,df1)
	Rxy = Cov2Sets(df1,df2)
	Ryx = Cov2Sets(df2,df1)
	Ryy = Cov2Sets(df2,df2)
	lines = (LinearAlgebra.inv(sqrt(Rxx))*Rxy*transpose(LinearAlgebra.inv(sqrt(Ryy))))
	p,L = eigen(lines*transpose(lines))
	p2,M = eigen(transpose(lines)*(lines))
	return L,M,p
end

#Main Extrinsic CCA function
function ECCA(df1,df2,n)
    L,M,p = findLMp(df1,df2)
    Ln = size(L)[2]
    Mn = size(M)[2]

    pred = transpose(transpose(inv(M))[:,Mn-n+1:Mn]*(p[Ln-n+1:Ln].*transpose(df1*L[:,Ln-n+1:Ln])))
    return pred
end


function ECCAusingLMp(df1,df2,n,L,M,p)
    Ln = size(L)[2]
    Mn = size(M)[2]
    pred = transpose(transpose(inv(M))[:,Mn-n+1:Mn]*(p[Ln-n+1:Ln].*transpose(df1*L[:,Ln-n+1:Ln])))
    return pred
end
