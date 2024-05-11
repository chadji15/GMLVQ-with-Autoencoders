classdef FCAE
% Only works for 28x28 images
    properties
        net
        encoderLayer
        decoder
        hiddenSize
    end
    
    methods
        function obj = FCAE(hiddenSize,images, maxEpochs, activation, plots)
           if ~exist('maxEpochs','var')
              maxEpochs=20;
           end
           if ~exist('activation','var')
              activation="sigmoid";
           end
           if ~exist('plots','var')
              plots="training-progress";
           end
           sz = size(images);
           imageSize = sz(1:3);
           activationLayer = sigmoidLayer;
           inputLayer = imageInputLayer(imageSize, "Normalization","none");
           ytrain = images;
           if activation == "tanh"
               inputLayer = imageInputLayer(imageSize, "Normalization","rescale-symmetric");
               %ytrain = rescale(ytrain,-1,1);
           end
           if activation == "tanh"
               activationLayer = tanhLayer;
           end
           % autoencoder layers
           layers = [ 
                inputLayer % 28x28x1
                convolution2dLayer(3,8, "stride", 2, "Padding","same") % 14x14x8
                convolution2dLayer(3,16,"Stride",1,"Padding","same") % 14x14x16
                convolution2dLayer(3,16, "stride", 2, "Padding","same") % 7x7x16
                convolution2dLayer(3,32,"Stride",1,"Padding","same") % 7x7x32
                convolution2dLayer(3,32, "stride", 2, "Padding","same") % 4x4x32
                convolution2dLayer(4, hiddenSize, "Stride",1) % 1x1xhiddenSize
                transposedConv2dLayer(4, 32, "stride", 1) % 4x4x32
                transposedConv2dLayer(3,32,"Stride",2,"Cropping","same") % 8x8x32
                convolution2dLayer(2,16,"Stride",1) % 7x7x716
                transposedConv2dLayer(3, 16, "stride", 2, "Cropping","same") % 14x14x16
                convolution2dLayer(3,8,"Stride",1,"Padding","same") % 14x14x8
                transposedConv2dLayer(3, imageSize(3), "stride", 2,"Cropping","same") % 28x28x1
                activationLayer
                regressionLayer
            ];
            
            
            % training hyperparameters
            options = trainingOptions('adam', ...
                'MaxEpochs',maxEpochs,...
                'InitialLearnRate',1e-3, ...
                'Verbose',false, ...
                'Plots',plots,...
                'MiniBatchSize',128);

            % train the network, use input as desired output
            
            net = trainNetwork(reshape(images, sz(1),sz(2),sz(3),[]), ...
                    reshape(ytrain, sz(1),sz(2),sz(3),[]), ...
                    layers, ...
                    options);
            
            obj.net = net;
            obj.hiddenSize = hiddenSize;
            % the point where the encoder ends
            obj.encoderLayer = 7;
            % isolate the decoder
            obj.decoder = assembleNetwork( ...
                [imageInputLayer([1 1 hiddenSize], "Normalization","none"); ...
                net.Layers(8:end)]);
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
            features = reshape(transpose(features), 1, 1,obj.hiddenSize,[]);
            output = obj.decoder.predict(features);
            % if obj.net.Layers(end-1).Type == "Tanh"
            %     output = rescale(output);
            % end
        end

        function [mmse, YTest] = test(obj,testImages, vis)
            if ~exist('vis','var')
              vis=true;
            end
            XTest = testImages;
            Xencoded = obj.encode(XTest);
            YTest = obj.decode(Xencoded);
            
            if vis
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
            end
            
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

