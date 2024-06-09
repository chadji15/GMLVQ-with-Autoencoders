classdef VAE
    %VAE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        encoder
        decoder
        hiddenSize
        activation
    end
    
    methods
        function obj = VAE(images, hiddenSize, varargin)
            %VAE Construct an instance of this class
            defaultMaxEpochs = 10;
            defaultPlots = 'training-progress';
            expectedPlots = {'training-progress', 'none'};
            defaultLearnRate = 1e-3;
            defaultActivation = 'sigmoid';
            expectedActivation = {'sigmoid', 'tanh'};  
            

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
            activation = p.Results.activation;
            % End of argument parsing
        
            sz = size(images);
            imageSize = sz(1:3);
              activationLayer = sigmoidLayer;
           inputLayer = imageInputLayer(imageSize, "Normalization","none");
           ytrain = images;
           if activation == "tanh"
               activationLayer = tanhLayer;
           end
           % This part takes care of compatibility of 28x28 and 32x32
           fistConvPadding = [0 0];
           if imageSize(1) == 32
               fistConvPadding = "same";
           end
            
            % Encoder layers
            layersE = [
                inputLayer % 28x28x1
                convolution2dLayer(3,8, "stride", 2, "Padding","same") % 14x14x8
                convolution2dLayer(3,16,"Stride",1,"Padding","same") % 14x14x16
                convolution2dLayer(3,16, "stride", 2, "Padding","same") % 7x7x16
                convolution2dLayer(3,32,"Stride",1,"Padding","same") % 7x7x32
                convolution2dLayer(3,32, "stride", 2, "Padding","same") % 4x4x32
                fullyConnectedLayer(2*hiddenSize)
                samplingLayer];
            
        
            numInputChannels = imageSize(3);
           % This is the size of the first set of image-like structures
           % after the bottleneck layer in order to apply transposed
           % convolution
           projectionSize = [4 4 32];
            
            % Decoder layers
            layersD = [
                featureInputLayer(hiddenSize)
                projectAndReshapeLayer(projectionSize)
                transposedConv2dLayer(3,32,"Stride",2,"Cropping","same") % 8x8x32
                convolution2dLayer(2,16,"Stride",1, "Padding",fistConvPadding) % 7x7x16 
                transposedConv2dLayer(3, 16, "stride", 2, "Cropping","same") % 14x14x16
                convolution2dLayer(3,8,"Stride",1,"Padding","same") % 14x14x8
                transposedConv2dLayer(3, imageSize(3), "stride", 2,"Cropping","same") % 28x28x1
                activationLayer];
            
            encoder = dlnetwork(layersE);
            decoder = dlnetwork(layersD);
            
            miniBatchSize = 128;
            XTrain = images;
            
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
                    [loss,gradientsE,gradientsD] = dlfeval(@modelLoss,encoder,decoder,X);
            
                    % Update learnable parameters.
                    [encoder,trailingAvgE,trailingAvgSqE] = adamupdate(encoder, ...
                        gradientsE,trailingAvgE,trailingAvgSqE,iteration,learnRate);
            
                    [decoder, trailingAvgD, trailingAvgSqD] = adamupdate(decoder, ...
                        gradientsD,trailingAvgD,trailingAvgSqD,iteration,learnRate);
            
                    % Update the training progress monitor. 
                    if plots == "training-progress"
                        recordMetrics(monitor,iteration,Loss=loss);
                        updateInfo(monitor,Epoch=epoch + " of " + maxEpochs);
                        monitor.Progress = 100*iteration/numIterations;
                    end
                end
            end

            obj.encoder = encoder;
            obj.decoder = decoder;
            obj.hiddenSize = hiddenSize;
            obj.activation = activation;

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
                z = predict(obj.encoder,X);
                % Extract and concatenate predictions.
                Y = cat(2,Y,extractdata(z));
            end
            Y = gather(Y)';
            
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
        %         z = predict(obj.decoder,X);
        %         % Extract and concatenate predictions.
        %         Y = cat(4,Y,extractdata(z));
        %     end
        % end

        function Y = decode(obj,codes)
            ZNew = dlarray(transpose(codes),"CB");
            Y = predict(obj.decoder,ZNew);
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
            
            YTest = modelPredictions(obj.encoder,obj.decoder,mbqTest);
            
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
                YTest = modelPredictions(obj.encoder,obj.decoder,mbqTest);
                msum = msum + sum(mse(XTest, YTest), "all");
            end
            mmse = msum/10;
        end 
    
        function YNew = generateNew(obj)
            numImages = 64;

            ZNew = randn(obj.hiddenSize,numImages);
            ZNew = dlarray(ZNew,"CB");
            
            YNew = predict(obj.decoder,ZNew);
            YNew = extractdata(YNew);
            
            figure
            I = imtile(YNew);
            imshow(I)
            title("Generated Images from noise")
        end
    end

end

