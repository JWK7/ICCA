#Imports CSV file and coverts it to dataframe
function GetData(filePath)
    df = CSV.read(filePath,DataFrame)
    return df
end

#Group data converted back to algebra data (euclidean based data)
function groupToalgData(intrinsicData)
	m = size(intrinsicData)[1]
	n = size(intrinsicData)[2]*3
	df  = zeros(m,n)
	for i in 1:m
		df[i,:] = getComp(intrinsicData[i,:,:,:])
	end
	return df
end

#Converts algebra data to group data
function LieData(df,groupList)
	m = size(df)[1]
	n = length(groupList)
	intrinsicData = zeros(m,n,3,3)
	
	for i in 1:m
		index = 1
		for j in 1:n
			if groupList[j] == "SO3" 
				intrinsicData[i,j,:,:] = Manifolds.exp_lie(SO3,[[0,df[i,index+2],-df[i,index+1]] [-df[i,index+2],0,df[i,index]] [df[i,index+1],-df[i,index],0]])
				index+=3
			elseif groupList[j] == "SO2"
				intrinsicData[i,j,:,:] = Manifolds.exp_lie(SO3,[[0,0,0] [0,0,df[i,index]] [0,-df[i,index],0]])
				index += 1
			end
			
		end
	end
	return intrinsicData
end

#Converts an individual algebra data to group data
function AlgToGroup(u,groupList)
	n = length(groupList)
	intrinsicData = zeros(n,3,3)
	
	index = 1
	for j in 1:n
		if groupList[j] == "SO3" 
			intrinsicData[j,:,:] = Manifolds.exp_lie(SO3,[[0,u[index+2],-u[index+1]] [-u[index+2],0,u[index]] [u[index+1],-u[index],0]])
			index+=3
		elseif groupList[j] == "SO2"
			intrinsicData[j,:,:] = Manifolds.exp_lie(SO3,[[0,0,0] [0,0,u[index]] [0,-u[index],0]])
			index += 1
		end
		
	end
return intrinsicData
end

#
function intrinsicToExtrinsicData(df)
	m = size(df)[1]
	n  = size(df)[2]
	dataFrame = zeros((m,n*3))
	for i in 1:m
		for j in 1:n
			dataFrame[i,:] = getComp(df[i,:,:,:])
		end
	end
	return dataFrame
end

function transferBackData(intrinsicData,mean)
	m = size(intrinsicData)[1]
	n = size(intrinsicData)[2]

	transferedData = zeros(m,n,3,3)
	for i in 1:m
		for j in 1:n
			transferedData[i,j,:,:] = (mean[j,:,:])*intrinsicData[i,j,:,:]
		end
	end
	return transferedData
end


function extrinsicTransfer(df)
	m = size(df)[1]
	n = size(df)[2]
	mean = zeros(n)
	transferedData = zeros(size(df))
	for i in 1:n
		mean[i] = sum(df[:,i])/length(df[:,i])
	end
	for i in 1:n
		transferedData[:,i] = df[:,i] .- mean[i]
	end
	return transferedData,mean
end


function transferData(intrinsicData)
	m = size(intrinsicData)[1]
	n = size(intrinsicData)[2]
	meanStates = zeros(n,3,3)
	for i in 1:n
		meanStates[i,:,:] = CalculateIntrinsicMean(intrinsicData[:,i,:,:])
	end
	transferedData = zeros(m,n,3,3)
	for i in 1:m
		for j in 1:n
			transferedData[i,j,:,:] = LinearAlgebra.inv(meanStates[j,:,:])*intrinsicData[i,j,:,:]
		end
	end
	return transferedData,meanStates
end


function extrinsicTransferBack(df,mean)
	m = size(df)[1]
	n = size(df)[2]
	transferedData = zeros(size(df))
	for i in 1:n
		transferedData[:,i] = df[:,i] .+ mean[i]
	end
	return transferedData
end
