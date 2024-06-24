classdef FCAE2
    %FCAE A class for containing a Fully Convolutional Autoencoder.
    % A fully convolutional autoencoder does not include any fully
    % connected layer. The architecture for this specific implementation:
    % Encoder: 6 convolution layers, 3 of which downsize
    % Decoder: 3 transposed convolution layers for upsampling and
    %  3 convolution layers that do not affect the output size.
    % Activation layer: sigmoid or tanh
    % only works for 28x28x? and 32x32x? images
    properties
        encoder % The number of the last layer of the encoder
        decoder % The last few layers of net, composing the decoder
        hiddenSize % The size of the bottleneck layer
        activation
    end
    
    methods
        function obj = FCAE2(images,hiddenSize,varargin)
            %FCAE
            % images: 4-D (S,S,C,B)
            % hiddenSize: positive integer, size of bottleneck layer
            % varargin: 
            %  activation: sigmoid or tanh
            %  plots: 'training-progress' for visual or 'none'
            %  learnRate: learning rate

            if nargin == 0
                return
            end

            % argument parsing
            defaultMaxEpochs = 10;
            defaultActivation = 'sigmoid';
            expectedActivation = {'sigmoid', 'tanh'};
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
            activation = p.Results.activation;
            plots = p.Results.plots;
            learnRate = p.Results.learnRate;

            % End of argument parsing.
          
           sz = size(images);
           imageSize = sz(1:3);
           activationLayer = sigmoidLayer;
           inputLayer = imageInputLayer(imageSize, "Normalization","none");
           ytrain = images;
           if activation == "tanh"
               activationLayer = tanhLayer;
           end
           % this part takes care of the 28x28 vs 32x32 compatibility
           fistConvPadding = [0 0];
           if imageSize(1) == 32
               fistConvPadding = "same";
           end
           %
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
                convolution2dLayer(2,16,"Stride",1, "Padding",fistConvPadding) % 7x7x16 
                transposedConv2dLayer(3, 16, "stride", 2, "Cropping","same") % 14x14x16
                convolution2dLayer(3,8,"Stride",1,"Padding","same") % 14x14x8
                transposedConv2dLayer(3, imageSize(3), "stride", 2,"Cropping","same") % 28x28x1
                activationLayer
            ];
            
            
            % training hyperparameters
            options = trainingOptions('adam', ...
                'MaxEpochs',maxEpochs,...
                'InitialLearnRate',learnRate, ...
                'Verbose',false, ...
                'Plots',plots,...
                'MiniBatchSize',128);

            % train the network, use input as desired output
            
            net = trainnet(reshape(images, sz(1),sz(2),sz(3),[]), ...
                    reshape(ytrain, sz(1),sz(2),sz(3),[]), ...
                    layers, ...
                    "mse", ...
                    options);
            obj.encoder = dlnetwork(net.Layers(1:7));
            obj.decoder = dlnetwork([imageInputLayer([1 1 hiddenSize], "Normalization","none"); ...
                net.Layers(8:end)]);
            obj.hiddenSize = hiddenSize;
            obj.activation = activation;
        end
        
        % use only the encoder from the trained network
        function features = encode(obj,images)
            sz = size(images);
            imr = reshape(images, sz(1), sz(2), sz(3),[]);
            %features = activations(obj.net, imr, obj.encoderLayer);
            features = minibatchpredict(obj.encoder,imr,"MiniBatchSize",32);
            features = transpose(reshape(features, obj.hiddenSize, []));
        end
        
        %use only the decoder from the trained network
        function output = decode(obj, features)
            features = reshape(transpose(features), 1, 1,obj.hiddenSize,[]);
            %output = obj.decoder.predict(features);
            output = minibatchpredict(obj.decoder,features,"MiniBatchSize",32);
        end

        function [mmse, YTest] = test(obj,testImages, vis)
            %test Evaluate the autoencoder on the test set
            % Shows some of the images and their reconstructions (if vis is
            % true)
            % Returns mse
            % 
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
            %generateNew Generate new images from noise
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

