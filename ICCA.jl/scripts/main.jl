begin
    using CSV
    using DataFrames
    using Manifolds
    using LinearAlgebra
    using Statistics
    using CSV
    using Distributions
    using DataFrames
    using MultivariateStats
	using MLJLinearModels
    using StatsPlots
    # using GLM
    using Plots

    SO3 = Manifolds.SpecialOrthogonal(3)
    include("../src/ICCA.jl")

    #DataFileNameList is a list of file name in the order of past train, future train, past test, and future test
    function main( trials = 3)



        TestErrorData = DataFrame(Manifold=[],Euclidean=[])
        TrainErrorData = DataFrame(Manifold=[],Euclidean=[])
        for j in 1:trials
            PastTrainDF = ICCA.GetData("data/pastTrainData"*string(j)*".csv")
            FutureTrainDF = ICCA.GetData("data/futureTrainData"*string(j)*".csv")
            PastTrainDF = select(PastTrainDF,[1,2,3,5,6,7,10,11,12,15,16,17,20,21,22])
            FutureTrainDF = select(FutureTrainDF,[1,2,3,5,6,7,10,11,12,15,16,17,20,21,22])
            groupList = ["SO3","SO3","SO3","SO3","SO3"]
            u,v,lossDf,meanFut,TstarSstarDF = ICCA.ICCA_Main(PastTrainDF,FutureTrainDF,groupList)
            CSV.write("output/Loss"*string(j)*".csv",lossDf)
            Plots.plot(lossDf.Loss, xlabel = "Iterations", ylabel = "ICCA Loss",legend = false)
            savefig("output/LossCurveTrainMultiJoint.png")
            Plots.scatter(TstarSstarDF.TStar,TstarSstarDF.SStar,xlabel="t*", ylabel="s*",aspect_ratio = :equal,legend = false)
            savefig("output/TstarvsSstarTrainMultiJoint.png")

            lasso = MLJLinearModels.GeneralizedLinearRegression()
            theta = MLJLinearModels.fit(lasso, reshape(TstarSstarDF.TStar,:,1),TstarSstarDF.SStar)
            SApproximated = TstarSstarDF.TStar*theta[1].+theta[2]
            predDataLieAlg = transpose(v * LinearAlgebra.transpose(SApproximated))
            predLieGroup  = ICCA.LieData(predDataLieAlg,groupList)
            finalPredictionGroup = ICCA.transferBackData(predLieGroup,meanFut)
            finalIntrinsicPred = ICCA.groupToalgData(finalPredictionGroup)

            extrinsicDataPast,extrinsicMeanPast = ICCA.extrinsicTransfer(PastTrainDF)
            extrinsicDataFuture,extrinsicMeanFuture = ICCA.extrinsicTransfer(FutureTrainDF)
            extPred = ICCA.ECCA(extrinsicDataPast,extrinsicDataFuture,6)
            finalExtrinsicPred = ICCA.extrinsicTransferBack(extPred,extrinsicMeanFuture)
            TrainErrorData = append!(TrainErrorData,DataFrame(Manifold=[ICCA.MSE(Matrix(FutureTrainDF),finalIntrinsicPred)/(size(finalIntrinsicPred)[1])],Euclidean=[ICCA.MSE(Matrix(FutureTrainDF),finalExtrinsicPred)/(size(finalIntrinsicPred)[1])]))


            PastTestDF = ICCA.GetData("data/pastTestData"*string(j)*".csv")
            FutureTestDF = ICCA.GetData("data/futureTestData"*string(j)*".csv")
            PastTestDF = select(PastTestDF,[1,2,3,5,6,7,10,11,12,15,16,17,20,21,22])
            FutureTestDF = select(FutureTestDF,[1,2,3,5,6,7,10,11,12,15,16,17,20,21,22])

            TestintrinsicDataInitial = ICCA.LieData(PastTestDF,groupList)
            TestintrinsicTransferedDataPast,TestmeanPast = ICCA.transferData(TestintrinsicDataInitial)
            TestintrinsicDataFut = ICCA.LieData(FutureTestDF,groupList)
            TestintrinsicTransferedDataFut,TestmeanFut = ICCA.transferData(TestintrinsicDataFut)
            TestextrinsicDataPast,TestextrinsicMeanPast = ICCA.extrinsicTransfer(PastTestDF)
            TestextrinsicDataFuture,TestextrinsicMeanFuture = ICCA.extrinsicTransfer(FutureTestDF)
            TestextPred = ICCA.ECCA(TestextrinsicDataPast,TestextrinsicDataFuture,1)
            TestfinalExtrinsicPred = ICCA.extrinsicTransferBack(TestextPred,TestextrinsicMeanFuture)
        
            ts = []
            for i in 1:size(TestintrinsicTransferedDataPast)[1]
                push!(ts,ICCA.findT(TestintrinsicTransferedDataPast[i,:,:,:],u))
            end
            
            TestSApproximated = ts*theta[1].+theta[2]
            TestpredDataLieAlg = transpose(v * LinearAlgebra.transpose(TestSApproximated))
            TestpredLieGroup  = ICCA.LieData(TestpredDataLieAlg,groupList)
            TestfinalPredictionGroup = ICCA.transferBackData(TestpredLieGroup,TestmeanFut)
            TestfinalIntrinsicPred = ICCA.groupToalgData(TestfinalPredictionGroup)

            TestErrorData = append!(TestErrorData,DataFrame(Manifold=[ICCA.MSE(Matrix(FutureTestDF),TestfinalIntrinsicPred)/(size(TestintrinsicTransferedDataPast)[1])],Euclidean = [ICCA.MSE(Matrix(FutureTestDF),TestfinalExtrinsicPred)/(size(TestintrinsicTransferedDataPast)[1])]))
            CSV.write("output/intrinsicPredictionTrain"*string(j)*".csv",DataFrame(finalIntrinsicPred,:auto))
            CSV.write("output/extrinsicPredictionTrain"*string(j)*".csv",DataFrame(finalExtrinsicPred,:auto))
            CSV.write("output/intrinsicPredictionTest"*string(j)*".csv",DataFrame(TestfinalIntrinsicPred,:auto))
            CSV.write("output/extrinsicPredictionTest"*string(j)*".csv",DataFrame(TestfinalExtrinsicPred,:auto))





        end
        CSV.write("output/TrainError.csv",TrainErrorData)
        CSV.write("output/TestError.csv",TestErrorData)

        split = repeat(["Train","Test"], inner = 2)
        StatsPlots.groupedbar(["ECCA","ICCA","ECCA","ICCA"],[1,mean(TrainErrorData.Manifold)/mean(TrainErrorData.Euclidean),1,mean(TestErrorData.Manifold)/mean(TestErrorData.Euclidean)],group=split,ylabel = "Avg. Error",color=[:LightBlue,:LightBlue,:NavyBlue,:Navyblue], guidefont=font(18),legendfont=font(13),xtickfont=font(13),grid=false)
        Plots.savefig("output/ErrorBar.png")


        Loss = ICCA.GetData("output/Loss1.csv")
        Loss2 = ICCA.GetData("output/Loss2.csv")
        Loss3 = ICCA.GetData("output/Loss3.csv")
        x = copy(Loss.Loss[1:100]/1000)
        x2 = copy(Loss2.Loss[1:100]/300)
        x3 = copy(Loss3.Loss[1:100]/300)
        xMean = (x+x2+x3)/3
        mat = [x x2 x3]
        xVar = zeros(size(mat)[1])
        for i in 1:size(mat)[1]
            xVar[i] = (var(mat[i,:]))
        end
        xMax = xMean+xVar
        xMin = xMean-xVar
        Plots.plot(xMin,fillrange = xMax,c =[:LightBlue],lw=2, fillalpha = 0.35,xlabel = "Iterations", ylabel = "ICCA Loss",label = "Confidence Bound",legend = :false)
        Plots.plot!(xMean,c =:Red,xlabel = "Iterations", lw=2, ylabel = "ICCA Loss", label = "Mean",legend = :topleft,
            xtickfont=font(13), 
            ytickfont=font(13),
            guidefont=font(18), 
            legendfont=font(13),grid=false)
        Plots.savefig("output/LossCurveTrainMultiJoint.png")
        return
    end


main(3)

end