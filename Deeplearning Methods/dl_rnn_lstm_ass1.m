%% Video Fade-Out Detection using LSTM in MATLAB

% Step 1: Read the Video
videoFile = '/MATLAB Drive/shot_test.mp4';  % Update with your video file path
v = VideoReader(videoFile);

% Initialize variables
frames = {};
frameInterval = 5;  % Extract every 5th frame for efficiency
count = 1;

% Extract frames from the video
while hasFrame(v)
    frame = readFrame(v);
    if mod(count, frameInterval) == 0
        frames{end + 1} = frame; %#ok<AGROW>
    end
    count = count + 1;
end
disp(['Total frames extracted: ', num2str(numel(frames))]);

%% Step 2: Prepare Data for Training and Validation
numFrames = numel(frames); % Total number of frames
X = []; % Feature sequences
Y = []; % Labels

for i = 1:numFrames
    img = frames{i}; 
    img = imresize(img, [224, 224]);  % Resize the image
    featureVector = img(:);  % Flatten the image into a feature vector
    
    % Normalize the feature vector
    featureVector = featureVector / 255;  % Scale pixel values to [0, 1]
    
    % Store the features
    X = [X, double(featureVector)];  % Ensure the feature vector is a double matrix
    
    % Define labels for each frame using the custom boundary detection function
    if boundaryDetected(i, frames)  % Improved labeling logic based on intensity threshold
        Y = [Y; 1];  % Fade-out detected
    else
        Y = [Y; 0];  % No fade-out
    end
end

% Convert labels to categorical for classification
Y = categorical(Y);  

% Display sizes for debugging
disp(['Size of X (features x samples): ', num2str(size(X))]); % Should be [features, samples]
disp(['Size of Y (samples): ', num2str(size(Y))]); % Should be [samples, 1]

% Split data into training and validation sets
cv = cvpartition(Y, 'HoldOut', 0.2); % 80% for training, 20% for validation
XTrain = X(:, training(cv));
YTrain = Y(training(cv));
XVal = X(:, test(cv));
YVal = Y(test(cv));

%% Step 3: Define and Train the LSTM Network
numFeatures = size(X, 1);  % Number of features
numClasses = 2;  % Two classes: fade-out or not

% Define LSTM layers
layers = [
    featureInputLayer(numFeatures, 'Name', 'input')
    lstmLayer(100, 'OutputMode', 'last', 'Name', 'lstm1') % LSTM layer with 100 hidden units
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')];

% Training options with custom metrics
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {XVal', YVal}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the LSTM network
try
    % Ensure XTrain is transposed to match the expected input shape
    XTrain = double(XTrain'); % Transpose XTrain to [samples, features]
    net = trainNetwork(XTrain, YTrain, layers, options);  
catch ME
    disp('Training failed:');
    disp(ME.message);  % Display error message if training fails
    return;  % Exit the function if training fails
end

%% Step 4: Evaluate Model Performance
% Predict on validation set
YPred = classify(net, double(XVal'));

% Calculate F1-score, precision, recall
confMat = confusionmat(YVal, YPred);
tp = confMat(2, 2);
fp = confMat(1, 2);
fn = confMat(2, 1);

precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1Score = 2 * (precision * recall) / (precision + recall);

disp('Performance Metrics on Validation Set:');
disp(['F1 Score: ', num2str(f1Score)]);
disp(['Precision: ', num2str(precision)]);
disp(['Recall: ', num2str(recall)]);

% Plot the confusion matrix
figure;
confusionchart(YVal, YPred);
title('Confusion Matrix for Validation Set');

% Plot Precision, Recall, and F1-Score
metrics = [precision, recall, f1Score];
figure;
bar(metrics);
set(gca, 'XTickLabel', {'Precision', 'Recall', 'F1-Score'});
ylabel('Score');
title('Model Performance Metrics on Validation Set');

%% Step 5: Predict Shot Boundaries in New Video
newVideoFile = '/MATLAB Drive/11.mp4';  % Update with your new video file path
vNew = VideoReader(newVideoFile);

newFrames = {};
count = 1;

while hasFrame(vNew)
    frame = readFrame(vNew);
    if mod(count, frameInterval) == 0
        newFrames{end + 1} = frame; %#ok<AGROW>
    end
    count = count + 1;
end
disp(['Total frames extracted for prediction: ', num2str(numel(newFrames))]);

% Prepare sequences for prediction
predX = [];
for i = 1:numel(newFrames)
    img = newFrames{i}; 
    img = imresize(img, [224, 224]);  % Resize the image
    featureVector = img(:);  % Flatten the image into a feature vector
    featureVector = featureVector / 255;  % Normalize the pixel values
    predX = [predX, double(featureVector)];  % Ensure the feature vector is a double matrix
end

% Reshape predX to match LSTM input shape for predictions
predX = double(predX');  % Transpose for predictions

% Check if the network exists before prediction
if exist('net', 'var')
    % Predict shot boundaries
    predictions = classify(net, predX);
    
    % Display predictions for debugging
    disp('Predictions:');
    disp(predictions);
    
    % Identify frames with fade-out
    fadeOutFrames = find(predictions == categorical(1));
    disp("Detected Fade-Out Frames at Indices:");
    disp(fadeOutFrames);
else
    disp('The network is not trained, so predictions cannot be made.');
end

%% Helper Function: Custom Boundary Detection Labeling
function isBoundary = boundaryDetected(frameIndex, frames)
    % Calculate the average pixel intensity of the frame
    img = frames{frameIndex};
    avgIntensity = mean(img(:)) / 255;  % Normalize intensity
    
    % Define a threshold for fade-out detection (adjust this based on your needs)
    threshold = 0.2;  % Adjust between 0 and 1 based on image data
    
    % Label as fade-out if below the threshold
    isBoundary = avgIntensity < threshold;
end
