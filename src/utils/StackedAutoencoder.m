classdef StackedAutoencoder
    
    properties
        hiddenSize
        encR
        encG
        encB
    end
    
    methods
        function obj = StackedAutoencoder(hiddenSize,images, maxEpochs)
           if ~exist('maxEpochs','var')
              maxEpochs=20;
           end
           obj.hiddenSize = hiddenSize;
           obj.encR = CustomAutoencoder22(hiddenSize, images(:,:,1,:), maxEpochs);
           obj.encG = CustomAutoencoder22(hiddenSize, images(:,:,2,:), maxEpochs);
           obj.encB = CustomAutoencoder22(hiddenSize, images(:,:,3,:), maxEpochs);
        end
        
        % use only the encoder from the trained network
        function features = encode(obj,images)
            featuresR = obj.encR.encode(images(:,:,1,:));
            featuresG = obj.encG.encode(images(:,:,2,:));
            featuresB = obj.encB.encode(images(:,:,3,:));
            features = cat(2, featuresR, featuresG, featuresB);
        end
        
        %use only the decoder from the trained network
        function output = decode(obj, features)
            featuresR = features(:,1:obj.hiddenSize);
            featuresG = features(:,obj.hiddenSize+1:2*obj.hiddenSize);
            featuresB = features(:,2*obj.hiddenSize+1:end);
            outputR = obj.encR.decode(featuresR);
            outputG = obj.encG.decode(featuresG);
            outputB = obj.encB.decode(featuresB);
            output = cat(3,outputR, outputG,outputB);
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
            numLatentChannels = obj.hiddenSize*3;
            
            ZNew = randn(numImages, numLatentChannels);
            
            YNew = obj.decode(ZNew);
            
            figure
            I = imtile(YNew);
            imshow(I)
            title("Generated Images from noise")
        end
    end
end

