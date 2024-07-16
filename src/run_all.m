% Custom auxiliary script for running multiple experiments with different
% configurations sequentially. I used this to automate running experiments
% on the Habrok cluster.

for activation = ["sigmoid"]
    for arch = ["CAE", "FCAE", "VAE"]
        settings.activation = activation;
        settings.learnRate = 1e-3;
        settings.doztr = true;
        settings.totalSteps = 30;
        settings.runs = 10;
        settings.percentage = 10; %for validation
        settings.arch = arch;
        settings.rescaleInput = false;
        
        %% MNIST
        settings.dataset = "MNIST";
        settings.savePath = "models/GMLVQ_" + arch + "_" + activation + ...
            "_" + settings.dataset + ".mat";
        settings.hiddenSize = 10;
        settings.numEpochs = 20;
        settings.classes = 0:9;

        savepath = run_full(settings);
        disp("Model for MNIST:" + ...
            savepath)

        %% FashionMNIST
        settings.dataset = "FashionMNIST";
        settings.savePath = "models/GMLVQ_" + arch + "_" + activation + ...
            "_" + settings.dataset  + ".mat";     
        
        settings.hiddenSize = 10;
        settings.numEpochs = 20;
        settings.classes = ["T-shirt/top", "Trouser", "Pullover", ...
            "Dress", "Coat","Sandal", "Shirt","Sneaker", "Bag", "Ankle boot"];
        savepath = run_full(settings);
        disp("Model for FashionMNIST:" + ...
            savepath)

        %% CIFAR-10
        settings.dataset = "CIFAR10";
        settings.savePath = "models/GMLVQ_" + arch + "_" + activation + ...
            "_" + settings.dataset  + ".mat";   
        
        settings.hiddenSize = 64;
        settings.numEpochs = 30;
        settings.classes = ["airplane", "automobile", "bird", "cat", ...
            "deer", "dog", "frog", "horse", "ship", "truck"];
        savepath = run_full(settings);
        disp("Model for CIFAR10:" + ...
            savepath)
    end
end