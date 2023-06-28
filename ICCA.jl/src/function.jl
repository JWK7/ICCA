#Creates Covariance matrix via euclidean space
function extrinsicCovarianceMatrix(x)
	
	matrix = transpose(x)*x
	matrix /= size(x)[1]
	return matrix
end

function extrinsicCovarianceMatrixWithIntrinsicTransferedData(x)
	n = size(x)[1]
	deltaS = getComp(x[1,:,:,:])*transpose(getComp(x[1,:,:,:]))
	summation = deltaS
	for j in 2:n
		deltaS = getComp(x[j,:,:,:])*transpose(getComp(x[j,:,:,:]))
		summation = summation + deltaS
	end
	summation = summation/(n)

	return summation
end

function getComp(df)
	comps = zeros(size(df)[1]*3)
	for i in 1:size(df)[1]
		comps[i*3] = Manifolds.log_lie(SO3,df[i,:,:])[2,1]
		comps[i*3-1] = Manifolds.log_lie(SO3,df[i,:,:])[1,3]
		comps[i*3-2] = Manifolds.log_lie(SO3,df[i,:,:])[3,2]
	end
	return comps
end

function optT(df,u)
	ts = zeros(size(df)[1])
	for i in 1:size(df)[1]
		ts[i] = findT(df[i,:,:,:],u)
	end
	ts
end

function CalculateIntrinsicMean(df)
	n = length(df[:,1,1])
	mew = 1.0*Matrix(LinearAlgebra.I, 3,3)

	deltaS = LinearAlgebra.inv(mew)*df[1,:,:]
	summation = Manifolds.log_lie(SO3,deltaS)
	for j in 2:n
		deltaS = LinearAlgebra.inv(mew)*df[j,:,:]
		summation = summation + Manifolds.log_lie(SO3,deltaS)
	end
	summation = summation/(n)
	deltaMew = Manifolds.exp_lie(SO3,summation)
	mew = mew*deltaMew
	return mew
	while LinearAlgebra.norm(Manifolds.log_lie(SO3,deltaMew)) > 0.000001
		deltaS = LinearAlgebra.inv(mew)*df[1,:,:]
		summation = Manifolds.log_lie(SO3,deltaS)
		for j in 2:n
			deltaS = LinearAlgebra.inv(mew)*df[j,:,:]
			summation = summation + Manifolds.log_lie(SO3,deltaS)
		end
		summation = summation/(n)
		deltaMew = Manifolds.exp_lie(SO3,summation)
		mew = mew*deltaMew
	end
	return mew
end



function findT(df,v)
	dis = zeros(size(df)[1]*3)
	for i in 1:size(df)[1]
		man = Manifolds.log_lie(SO3,df[i,:,:])
		dis[i*3] =man[3,2]
		dis[i*3-1] = man[1,3]
		dis[i*3-2] = man[2,1]
	end
	a = LinearAlgebra.tr(LinearAlgebra.Transpose(dis)*v)
	b = LinearAlgebra.tr(LinearAlgebra.Transpose(v)*v)

	t = a/b
	t= 0
	increase = 1
	if Error(df,v,t + 0.01) > Error(df,v,t)
		increase = -1
	end
	
	for i in 1:300
		currentE = Error(df,v,t)
		stepE = Error(df,v,t + increase *0.01)
		if currentE <= stepE
			# println(i)
			# println(currentE)
			break
		end
		t += increase * 0.01
	end
	return t
end


function Error(Data,IPCA,t1)
	IPCAError = 0
	tv1 = t1*IPCA
	for j in 1:size(Data)[1]
		man = Manifolds.exp_lie(SO3,[[0,tv1[j*3],-tv1[j*3-1]] [-tv1[j*3],0,tv1[j*3-2]] [tv1[j*3-1],-tv1[j*3-2],0]])
		man2 = Manifolds.log_lie(SO3,(LinearAlgebra.inv(man) * Data[j,:,:]))
		# man = abs(Manifolds.log_lie(SO3,LinearAlgebra.inv(Manifolds.exp_lie(SO3,[[[0,tv1[j]] [-tv1[j],0]]))*Data[j,:,:])[2,1]])
		IPCAError += abs(man2[3,2]) + abs(man2[1,3]) + abs(man2[2,1])
	end
	# println(IPCAError)
	return IPCAError
end

function getDistance(tu,df)
	d = zeros(size(df)[1])
	for i in 1:size(df)[1]

		man = Manifolds.exp_lie(SO3,[[0,tu[i*3],-tu[i*3-1]] [-tu[i*3],0,tu[i*3-2]] [tu[i*3-1],-tu[i*3-2],0]])
		man2 = Manifolds.log_lie(SO3,(LinearAlgebra.inv(man) * df[i,:,:]))
		d[i] = abs(man2[3,2]) + abs(man2[1,3]) + abs(man2[2,1])
		# man =  Manifolds.log_lie(SO3,LinearAlgebra.inv(Manifolds.exp_lie(SO3,[[0,tu[i]] [-tu[i],0]]))*df[i,:,:])
		# d[i] = man[3,2] + man[1,3] + man[2,1]
	end
	return d
end



function GetPrincipalComponent(df)

	lr = 1
	min = Inf
	u = LinearAlgebra.normalize(ones(size(df)[2]*3))
	for j in 1:1000
		summation = zeros(size(df)[2]*3)
		for k in 1:size(df)[2]*3
			for i in 1:size(df)[1]
				u2 = copy(u)
				u2[k] += lr
				u2 = LinearAlgebra.normalize(u2)
				t = findT(df[i,:,:,:],u2)
				summation[k] += Error(df[i,:,:,:],u2,t)
			end
		end 
	




		summation2 = zeros(size(df)[2]*3)
		for k in 1:size(df)[2]*3
			for i in 1:size(df)[1]
				u2 = copy(u)
				u2[k] -= lr
				u2 = LinearAlgebra.normalize(u2)
				t = findT(df[i,:,:,:],u2)
				summation2[k] += Error(df[i,:,:,:],u2,t)
			end
		end
		if minimum(summation) <= minimum(summation2)
			if min <= minimum(summation)
				lr /= 2
			else
				u[argmin(summation)] += lr
				min = (minimum(summation))
			end
		else 
			if min <= minimum(summation2)
				lr /= 2
			else
				u[argmin(summation2)] -= lr
				min = minimum(summation2)
			end
		end

		if lr < 0.000001
			break
		end
		u = LinearAlgebra.normalize(u)
	end

	return u
end






#Flip matrix columns direction
function flip(df)
	data = zeros(size(df))
	for i in 1:size(df)[2]
		data[:,i] = df[:,size(df)[2]+1-i]
	end
	return data
end

#Covariance set between two datasets
function Cov2Sets(df1,df2)
	matrix = transpose(df1)*df2
	matrix /= size(df1)[1]	
	return matrix
end


#MSE Error Calc
function MSE(prediction,y)
    return sum(abs.(y.-prediction).^2)
end


