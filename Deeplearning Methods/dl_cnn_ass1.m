% Load Video and Extract Frames
videoFilePath = '/MATLAB Drive/11.mp4'; % Path to your video
video = VideoReader(videoFilePath);
frames = {};

while hasFrame(video)
    frames{end+1} = readFrame(video);
end

% Check the number of frames extracted
numFrames = length(frames);
if numFrames < 10 % Ensure there are enough frames to train
    error('Not enough frames in the video for training.');
end

% Load or Generate Labels
if isfile('labels.mat')
    load('labels.mat');
else
    labels = randi([0, 1], [numFrames, 1]); % Random binary labels for testing
    save('labels.mat', 'labels');
end

labels = categorical(labels);

% Adjust labels to match frames
if length(labels) ~= numFrames
    labels = labels(1:numFrames); % Adjust labels if too many
    warning('Labels adjusted to match number of frames.');
end

% Prepare Input Data
inputSize = [224 224 3];
X = zeros([inputSize, numFrames], 'single'); % Initialize 4D array for images

for i = 1:numFrames
    resizedFrame = imresize(frames{i}, inputSize(1:2)); % Resize frame
    X(:, :, :, i) = single(resizedFrame); % Store resized frame
end

% Define CNN Layers
layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'ValidationFrequency', 5);

% Train the Model
net = trainNetwork(X, labels, layers, options);

% Load Fade-Out Frame
fadeOutFramePath = '/MATLAB Drive/frame.png'; % Path to your fade-out frame
fadeOutFrame = imread(fadeOutFramePath);
fadeOutFrameResized = imresize(fadeOutFrame, inputSize(1:2)); % Resize to match input size
fadeOutFrameSingle = single(fadeOutFrameResized); % Convert to single precision

% Reset video reader for new predictions
video = VideoReader(videoFilePath);
frameIndex = 1; % Initialize frame index

% Initialize counter and array for detected frames
detectedFrameCount = 0;
detectedFrames = []; % Array to store indices of detected frames
predictions = zeros(numFrames, 1); % Array to store predictions

% Detect Fade-Out
while hasFrame(video)
    currentFrame = readFrame(video);
    if frameIndex > numFrames
        break; % Exit if all frames are processed
    end
    currentFrameResized = imresize(currentFrame, inputSize(1:2));
    currentFrameSingle = single(currentFrameResized);

    % Calculate similarity metrics
    mse = immse(currentFrameSingle, fadeOutFrameSingle);
    ssimValue = ssim(currentFrameSingle, fadeOutFrameSingle);
    
    % Print MSE and SSIM values
    disp(['Frame ', num2str(frameIndex), ': MSE = ', num2str(mse), ', SSIM = ', num2str(ssimValue)]);

    % Define thresholds
    thresholdMSE = 0.001; % Adjust MSE threshold
    thresholdSSIM = 0.95; % Adjust SSIM threshold

    % Check for shot boundary detection
    if mse < thresholdMSE || ssimValue > thresholdSSIM
        detectedFrameCount = detectedFrameCount + 1; % Increment the count
        detectedFrames(end+1) = frameIndex; % Store the frame index
        predictions(frameIndex) = 1; % Mark as detected
        disp(['Frame ', num2str(frameIndex), ': Shot boundary detected (fade-out).']);
        imshow(currentFrameResized); % Show the detected frame
        title(['Detected Frame: ', num2str(frameIndex)]);
        pause(1); % Pause for a second to view
    else
        predictions(frameIndex) = 0; % Mark as not detected
        disp(['Frame ', num2str(frameIndex), ': No shot boundary detected.']);
    end

    frameIndex = frameIndex + 1; % Move to the next frame
end

% Display total detected frames
disp(['Total fade-out frames detected: ', num2str(detectedFrameCount)]);
disp('Indices of detected frames:');
disp(detectedFrames);

% Calculate Evaluation Metrics
TP = sum((predictions == 1) & (labels == '1')); % True Positives
FP = sum((predictions == 1) & (labels == '0')); % False Positives
TN = sum((predictions == 0) & (labels == '0')); % True Negatives
FN = sum((predictions == 0) & (labels == '1')); % False Negatives

% Precision, Recall, F1 Score
precision = TP / (TP + FP);
recall = TP / (TP + FN);
F1 = 2 * (precision * recall) / (precision + recall);

% Confusion Matrix
confusionMatrix = [TP, FP; FN, TN];

% Display Results
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', F1);
disp('Confusion Matrix:');
disp(confusionMatrix);

% Plotting the results
figure; % Create a new figure for the confusion matrix

% Confusion Matrix Heatmap
h = heatmap(confusionMatrix, 'XLabel', 'Predicted', 'YLabel', 'Actual', ...
    'ColorbarVisible', 'off', 'CellLabelColor', 'none', ...
    'XDisplayLabels', {'No Fade-Out', 'Fade-Out'}, ...
    'YDisplayLabels', {'No Fade-Out', 'Fade-Out'}, ...
    'FontSize', 12); % Increased font size for clarity

% Set the title for the heatmap
h.Title = 'Confusion Matrix'; % Set title using heatmap Title property
h.FontSize = 12; % Set font size for the heatmap

% Create a new figure for precision, recall, and F1 score
figure;

% F1 Score, Precision, Recall Plot
bar([precision, recall, F1], 'FaceColor', [0.2, 0.6, 0.8]); % Set bar color
set(gca, 'XTickLabel', {'Precision', 'Recall', 'F1 Score'}, 'FontSize', 12); % Set labels and font size
ylabel('Score', 'FontSize', 12);
ylim([0 1]);
title('Performance Metrics', 'FontSize', 14); % Increased title font size
grid on; % Add grid for better readability

sgtitle('Shot Boundary Detection Metrics', 'FontSize', 16); % Set super title font size
