classdef VAE
    %VAE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        netE
        netD
        hiddenSize
    end
    
    methods
        function obj = VAE(netE, netD, hiddenSize)
            %VAE Construct an instance of this class
            %   Detailed explanation goes here
            obj.netE = netE;
            obj.netD = netD;
            obj.hiddenSize = hiddenSize;
        end
        
        function Y = encode(obj,images)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            dsTest = arrayDatastore(images,IterationDimension=4);
            numOutputs = 1;
            miniBatchSize = 128;
            
            mbq = minibatchqueue(dsTest,numOutputs, ...
                MiniBatchSize = miniBatchSize, ...
                MiniBatchFcn=@preprocessMiniBatch, ...
                MiniBatchFormat="SSCB");

            Y = [];

            % Loop over mini-batches.
            while hasdata(mbq)
                X = next(mbq);
            
                % Forward through encoder.
                z = predict(obj.netE,X);
                % Extract and concatenate predictions.
                Y = cat(2,Y,extractdata(z));
            end
            Y = gather(Y);
        end

        % function Y = decode(obj,codes)
        % 
        %     %METHOD1 Summary of this method goes here
        %     %   Detailed explanation goes here
        %     dsTest = arrayDatastore(codes,IterationDimension=1);
        %     numOutputs = 1;
        %     miniBatchSize = 128;
        % 
        %     mbq = minibatchqueue(dsTest,numOutputs, ...
        %         MiniBatchSize = miniBatchSize, ...
        %         MiniBatchFcn=@(dataX)cat(1,dataX{:}), ...
        %         MiniBatchFormat="BS");
        % 
        %     Y = [];
        % 
        %     % Loop over mini-batches.
        %     while hasdata(mbq)
        %         X = next(mbq);
        % 
        %         % Forward through encoder.
        %         z = predict(obj.netD,X);
        %         % Extract and concatenate predictions.
        %         Y = cat(4,Y,extractdata(z));
        %     end
        % end

        function Y = decode(obj,codes)
            ZNew = dlarray(transpose(codes),"CB");
            Y = predict(obj.netD,ZNew);
            Y = extractdata(Y);
        end
    
        function mmse = test(obj, testImages)
            XTest = testImages;
            dsTest = arrayDatastore(XTest,IterationDimension=4);
            numOutputs = 1;
            miniBatchSize = 128;
            
            mbqTest = minibatchqueue(dsTest,numOutputs, ...
                MiniBatchSize = miniBatchSize, ...
                MiniBatchFcn=@preprocessMiniBatch, ...
                MiniBatchFormat="SSCB");
            
            YTest = modelPredictions(obj.netE,obj.netD,mbqTest);
            
            err = mean((single(XTest)-YTest).^2,[1 2 3]);
            figure
            histogram(err)
            xlabel("Error")
            ylabel("Frequency")
            title("Test Data")
            
            figure
            subplot(1,2,1)
            imshow(imtile(XTest(:,:,:,1:64)));
            title("Test images");
            subplot(1,2,2)
            imshow(imtile(YTest(:,:,:,1:64)));
            title("VAE reconstruction");
            
            msum = 0;
            for i = 1:10
                mbqTest.reset();
                YTest = modelPredictions(obj.netE,obj.netD,mbqTest);
                msum = msum + sum(mse(XTest, YTest), "all");
            end
            mmse = msum/10;
        end 
    
        function YNew = generateNew(obj)
            numImages = 64;

            ZNew = randn(obj.hiddenSize,numImages);
            ZNew = dlarray(ZNew,"CB");
            
            YNew = predict(obj.netD,ZNew);
            YNew = extractdata(YNew);
            
            figure
            I = imtile(YNew);
            imshow(I)
            title("Generated Images from noise")
        end
    end

    methods (Static)
        % Static methods acting as "constructor overloads"
        %function obj = trainVAE(hiddenSize,trainingImages, maxEpochs, learnRate)
        function obj = trainVAE(trainingImages, hiddenSize, varargin)
            defaultMaxEpochs = 10;
            defaultPlots = 'training-progress';
            expectedPlots = {'training-progress', 'none'};
            defaultLearnRate = 1e-3;
    
            p = inputParser;
            validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
            addRequired(p,'images');
            addRequired(p,"hiddenSize", validScalarPosNum);
            addOptional(p,"maxEpochs", defaultMaxEpochs, validScalarPosNum);
            addParameter(p,"activation",defaultActivation, ...
                @(x) any(validatestring(x,expectedActivation)))
            addParameter(p,"plots",defaultPlots, ...
                @(x) any(validatestring(x,expectedPlots)))
            addParameter(p,"learnRate",defaultLearnRate);
    
            parse(p,images, hiddenSize,varargin{:});

            maxEpochs = p.Results.maxEpochs;
            plots = p.Results.plots;
            learnRate = p.Results.learnRate;
        
            sz = size(trainingImages);
            imageSize = sz(1:3);
            
            % Encoder layers
            layersE = [
                imageInputLayer(imageSize,Normalization="none")
                convolution2dLayer(3,32,Padding="same",Stride=2) % 28x28x1 => 14x14x32
                reluLayer
                convolution2dLayer(3,64,Padding="same",Stride=2)% 14x14x32 => 7x7x64
                reluLayer
                fullyConnectedLayer(2*hiddenSize) % 7x7x64 => 1x64
                samplingLayer];
            
            projectionSize = [7 7 64];
            if imageSize(1) == 28
                projectionSize = [7 7 64];
            elseif imageSize(1) == 32
                projectionSize = [8 8 64];
            end
        
            numInputChannels = imageSize(3);
            
            % Decoder layers
            layersD = [
                featureInputLayer(hiddenSize)
                projectAndReshapeLayer(projectionSize)
                transposedConv2dLayer(3,64,Cropping="same",Stride=2) % 7x7x64 => 14x14x32
                reluLayer
                transposedConv2dLayer(3,32,Cropping="same",Stride=2) % 14x14x32 => 28x28x32
                reluLayer
                transposedConv2dLayer(3,numInputChannels,Cropping="same") % 28x28x32 => 28x28x1
                sigmoidLayer];
            
            netE = dlnetwork(layersE);
            netD = dlnetwork(layersD);
            
            miniBatchSize = 128;
            XTrain = trainingImages;
            
            dsTrain = arrayDatastore(XTrain,IterationDimension=4);
            numOutputs = 1;
            
            mbq = minibatchqueue(dsTrain,numOutputs, ...
                MiniBatchSize = miniBatchSize, ...
                MiniBatchFcn=@preprocessMiniBatch, ...
                MiniBatchFormat="SSCB", ...
                PartialMiniBatch="discard");
            
            % Initialize the parameters for the Adam solver.
            trailingAvgE = [];
            trailingAvgSqE = [];
            trailingAvgD = [];
            trailingAvgSqD = [];
            
            % Calculate the total number of iterations for the training progress monitor
            numObservationsTrain = size(XTrain,4);
            numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
            numIterations = maxEpochs * numIterationsPerEpoch;
            
            %Initialize the training progress monitor.
            if plots == "training-progress"
            monitor = trainingProgressMonitor( ...
                Metrics="Loss", ...
                Info="Epoch", ...
                XLabel="Iteration");
            end
            
            epoch = 0;
            iteration = 0;
            
            % Custom training loop
            % Loop over epochs.
            while epoch < maxEpochs && ~monitor.Stop
                epoch = epoch + 1;
            
                % Shuffle data.
                shuffle(mbq);
            
                % Loop over mini-batches.
                while hasdata(mbq) && ~monitor.Stop
                    iteration = iteration + 1;
            
                    % Read mini-batch of data.
                    X = next(mbq);
            
                    % Evaluate loss and gradients.
                    [loss,gradientsE,gradientsD] = dlfeval(@modelLoss,netE,netD,X);
            
                    % Update learnable parameters.
                    [netE,trailingAvgE,trailingAvgSqE] = adamupdate(netE, ...
                        gradientsE,trailingAvgE,trailingAvgSqE,iteration,learnRate);
            
                    [netD, trailingAvgD, trailingAvgSqD] = adamupdate(netD, ...
                        gradientsD,trailingAvgD,trailingAvgSqD,iteration,learnRate);
            
                    % Update the training progress monitor. 
                    if plots == "training-progress"
                        recordMetrics(monitor,iteration,Loss=loss);
                        updateInfo(monitor,Epoch=epoch + " of " + maxEpochs);
                        monitor.Progress = 100*iteration/numIterations;
                    end
                end
            end

            obj = VAE(netE, netD, hiddenSize);
        end
    end
end

