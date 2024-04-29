classdef LabelTransformer
    %LabelTransformer: This is a simple class for tranforming (sparse)
    %labels to to the range 1-C. 
    %   C is the number of unique labels
    %   This helps us use our datasets with the GMLVQ toolbox easier.

    properties
        labelMap = dictionary % Stores the mapping from the original labels to the new ones
        reverseMap = dictionary % Stores the reverse mapping
    end

    methods
        function obj = LabelTransformer(classes)
            %Construct an instance of this class
            %   A new transformer is created based on the provided classes
            %   classes: a 1xC array with each unique possible class
            for i=1:length(classes)
                obj.labelMap(classes(i)) = i;  
            end
            obj.reverseMap = dictionary(obj.labelMap.values, obj.labelMap.keys);
        end

        function newLabels = transform(obj, labels)
            %Transform the given labels based on the initialized class map
            %   obj: a LabelTransformer object
            %   labels: a 1xN matrix with values from the original possible
            %   class
            %   Returns : a 1xN matrix with the new mapped values 
            newLabels = arrayfun(@(x) obj.labelMap(x), labels);
        end

        function originalLabels = reverse(obj, labels)
            %Revert the tranformed labels to their original values
            %   obj: a labels transformer object
            %   labels: a 1xN matrix with the mapped labels
            %   Returns: a 1xN matrix with the corresponding original
            %   labels
            originalLabels = arrayfun(@(x) obj.reverseMap(x), labels);
        end
    end
end