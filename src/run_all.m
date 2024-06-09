for activation = ["tanh", "sigmoid"]
    for arch = ["VAE"]
        settings.activation = activation;
        settings.learnRate = 1e-3;
        settings.doztr = true;
        settings.totalSteps = 30;
        settings.runs = 1;
        settings.percentage = 10; %for validation
        settings.arch = arch;
        settings.rescaleInput = false;
        
        %% MNIST
        settings.dataset = "MNIST";
        settings.savePath = "models/GMVLQ_" + arch + "_" + activation + ...
            "_" + settings.dataset + ".mat";
        settings.hiddenSize = 10;
        settings.numEpochs = 20;
        
        savepath = run_full(settings);
        disp("Model for MNIST:" + ...
            savepath)
        
        %% FashionMNIST
        settings.dataset = "FashionMNIST";
        settings.savePath = "models/GMVLQ_" + arch + "_" + activation + ...
            "_" + settings.dataset  + ".mat";     
        savepath = run_full(settings);
        disp("Model for FashionMNIST:" + ...
            savepath)
    end
end