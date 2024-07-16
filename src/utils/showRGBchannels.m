function [outputArg1,outputArg2] = showRGBchannels(img)
% showRGBchannels
% Utility function to display the three channels of an RGB image
% separately.
rgbImage = img;
% Extract color channels.
redChannel = rgbImage(:,:,1); % Red channel
greenChannel = rgbImage(:,:,2); % Green channel
blueChannel = rgbImage(:,:,3); % Blue channel
% Create an all black channel.
allBlack = zeros(size(rgbImage, 1), size(rgbImage, 2), 'double');
 
% Create color versions of the individual color channels.
just_red = cat(3, redChannel, allBlack, allBlack);
just_green = cat(3, allBlack, greenChannel, allBlack);
just_blue = cat(3, allBlack, allBlack, blueChannel);

% Display them all.
subplot(1, 4, 1);
imshow(rgbImage);
subplot(1, 4, 2);
imshow(just_red);
subplot(1, 4, 3);
imshow(just_green)
subplot(1, 4, 4);
imshow(just_blue);
end