function ICCA_Main(df,df2,groupList)
	intrinsicDataInitial = LieData(df,groupList)
	intrinsicTransferedDataPast,meanPast = transferData(intrinsicDataInitial)
	intrinsicDataFut = LieData(df2,groupList)
	intrinsicTransferedDataFut,meanFut = transferData(intrinsicDataFut)
	u,v,lossDf = ICCACanonicalPair(intrinsicTransferedDataPast,intrinsicTransferedDataFut,groupList)
	Tstar = optT(intrinsicTransferedDataPast,u)
	Sstar = optT(intrinsicTransferedDataFut,v)
	TstarSstarDF = DataFrame(TStar=Tstar,SStar=Sstar)
	return u,v,lossDf,meanFut,TstarSstarDF
end


function ICCACanonicalPair(df,df2,groupList)
	n = size(df)[2]*3
	m = size(df)[1]
	lr = 1
	mini = 0
	before = 0
	count = 0
	u = GetPrincipalComponent(df)
	v = GetPrincipalComponent(df2)
	lossDf = DataFrame(Loss=Float64[])
	for i in 1:m
		mini += Error(df[i,:,:,:],u,findT(df[i,:,:,:],u))
		mini += Error(df2[i,:,:,:],v,findT(df2[i,:,:,:],v))
		mini += distanceBetweensvtu(findT(df[i,:,:,:],u) * u,findT(df2[i,:,:,:],v) * v,groupList)
	end
	
	for j in 1:1000
		summation = zeros(n*2)
		for k in 1:n*2
			for i in 1:m
				u2 = copy(u)
				v2 = copy(v)
				if k <= n 
					u2[k] += lr
					u2 = LinearAlgebra.normalize(u2)
				else
					v2[k-n] += lr
					v2 = LinearAlgebra.normalize(v2)
				end
				t = findT(df[i,:,:,:],u2)
				s = findT(df2[i,:,:,:],v2)
				summation[k] += Error(df[i,:,:,:],u2,t)
				summation[k] += Error(df2[i,:,:,:],v2,s)
				summation[k] += distanceBetweensvtu(v2*s,u2*t,groupList)
			end
		end 
		
		summation2 = zeros(n*2)
		for k in 1:n*2
			for i in 1:m
				u2 = copy(u)
				v2 = copy(v)
				if k <= n 
					u2[k] -= lr
					u2 = LinearAlgebra.normalize(u2)
				else
					v2[k-n] -= lr
					v2 = LinearAlgebra.normalize(v2)
				end
				t = findT(df[i,:,:,:],u2)
				s = findT(df2[i,:,:,:],v2)
				summation2[k] += Error(df[i,:,:,:],u2,t)
				summation2[k] += Error(df2[i,:,:,:],v2,s)
				summation2[k] += distanceBetweensvtu(v2*s,u2*t,groupList)
			end
		end 
		
		if minimum(summation) <= minimum(summation2)
			if mini <= minimum(summation)
				lr /= 2
			else
				k = argmin(summation)
				if k <= n
					u[argmin(summation)] += lr
				else
					v[argmin(summation)-n] += lr
				end
				mini = (minimum(summation))
				# push!(lossDf,[mini])
			end
		else 
			if mini <= minimum(summation2)
				lr /= 2
			else
				k = argmin(summation2)
				if k <= n
					u[argmin(summation2)] -= lr
				else
					v[argmin(summation2)-n] -= lr
				end
				mini = minimum(summation2)
				# push!(lossDf,[mini])
			end
		end
	
		push!(lossDf,[mini])
		u = LinearAlgebra.normalize(u)
		v = LinearAlgebra.normalize(v)
		if lr < 0.000001
			break
		end
	end
	return u,v,lossDf
end





function distanceBetweensvtu(sv,tu,groupList)
	SV =  AlgToGroup(sv,groupList)
	TU = AlgToGroup(tu,groupList)
	distance = 0
	for i in size(SV)[1]
		dis =  Manifolds.log_lie(SO3,LinearAlgebra.inv(Manifolds.exp_lie(SO3,SV[i,:,:]))*Manifolds.exp_lie(SO3,TU[i,:,:]))
		distance += abs(dis[2,1]) + abs(dis[1,3]) + abs(dis[3,2])
	end
	return distance
end