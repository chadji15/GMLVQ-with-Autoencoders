classdef CustomAutoencoder22
    
    properties
        net
        encoderLayer
        decoder
        hiddenSize
    end
    
    methods
        function obj = CustomAutoencoder22(hiddenSize,images, maxEpochs)
           if ~exist('maxEpochs','var')
              maxEpochs=20;
           end
           sz = size(images);
           imageSize = sz(1:3);
           if sz(1) == 28
               projectionSize = [7 7 64];
           elseif sz(2) == 32
               projectionSize = [8 8 64];
           end
           % autoencoder layers
           layers = [ 
                imageInputLayer(imageSize,Normalization="none")
                convolution2dLayer(3,32,Padding="same",Stride=1)
                reluLayer
                convolution2dLayer(3,32,Padding="same",Stride=2) % 28x28x1 => 14x14x32
                reluLayer
                convolution2dLayer(3,64,Padding="same",Stride=1)% 14x14x32 => 7x7x64
                reluLayer
                convolution2dLayer(3,64,Padding="same",Stride=2)% 14x14x32 => 7x7x64
                reluLayer
                fullyConnectedLayer(hiddenSize) % 7x7x64 => 1x32
                projectAndReshapeLayer(projectionSize)
                transposedConv2dLayer(3,64,Cropping="same",Stride=1) % 7x7x64 => 14x14x32
                reluLayer
                transposedConv2dLayer(3,64,Cropping="same",Stride=2) % 7x7x64 => 14x14x32
                reluLayer
                transposedConv2dLayer(3,32,Cropping="same",Stride=1) % 14x14x32 => 28x28x32
                reluLayer
                transposedConv2dLayer(3,32,Cropping="same",Stride=2) % 14x14x32 => 28x28x32
                reluLayer
                transposedConv2dLayer(3,imageSize(3),Cropping="same") % 28x28x32 => 28x28x1
                sigmoidLayer
                regressionLayer
            ];
            
            % training hyperparameters
            options = trainingOptions('adam', ...
                'MaxEpochs',maxEpochs,...
                'InitialLearnRate',1e-3, ...
                'Verbose',false, ...
                'Plots','training-progress',...
                'MiniBatchSize',128);

            % train the network, use input as desired output
            net = trainNetwork(reshape(images, sz(1),sz(2),sz(3),[]), ...
                    reshape(images, sz(1),sz(2),sz(3),[]), ...
                    layers, ...
                    options);
            
            obj.net = net;
            obj.hiddenSize = hiddenSize;
            % the point where the encoder ends
            obj.encoderLayer = 10;
            % isolate the decoder
            obj.decoder = assembleNetwork( ...
                [featureInputLayer(hiddenSize); ...
                net.Layers(11:end)]);
        end
        
        % use only the encoder from the trained network
        function features = encode(obj,images)
            sz = size(images);
            imr = reshape(images, sz(1), sz(2), sz(3),[]);
            features = activations(obj.net, imr, obj.encoderLayer);
            features = transpose(reshape(features, obj.hiddenSize, []));
        end
        
        %use only the decoder from the trained network
        function output = decode(obj, features)
            output = obj.decoder.predict(features);
        end

        function [mmse, YTest] = test(obj,testImages)
            XTest = testImages;
            Xencoded = obj.encode(XTest);
            YTest = obj.decode(Xencoded);
            
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
            title("Autoencoder reconstruction");
            
            msum = 0;
            for i = 1:10
                YTest = obj.decode(Xencoded);
                msum = msum + sum(mse(XTest, YTest), "all");
            end
            mmse = msum/10;
        end
    
        function YNew = generateNew(obj)
            numImages = 64;
            numLatentChannels = obj.hiddenSize;
            
            ZNew = randn(numImages, numLatentChannels);
            
            YNew = obj.decode(ZNew);
            
            figure
            I = imtile(YNew);
            imshow(I)
            title("Generated Images from noise")
        end
    end
end

