% Load video
video = VideoReader('/MATLAB Drive/11.mp4');
numFrames = video.NumFrames;

% Parameters
threshold = 3000; % Threshold to detect significant scene changes (high spikes)
minDifference = 500; % Minimum difference to ignore minor changes
spikeDifference = 20000; % Minimum spike difference to consider major scene change

% Initialize arrays to store histogram differences
histDiffs = zeros(numFrames - 1, 1);

% Loop through frames to calculate histogram differences
for k = 1:numFrames-1
    % Read and process the current and next frames
    frame1 = rgb2gray(read(video, k));
    frame2 = rgb2gray(read(video, k + 1));
    
    % Calculate histograms
    [P, ~] = imhist(frame1);
    [R, ~] = imhist(frame2);
    
    % Compute absolute histogram difference
    histDiff = sum(abs(P - R));
    
    % Only store if difference exceeds minDifference
    if histDiff > minDifference
        histDiffs(k) = histDiff;
        
        % If histDiff is a major spike compared to the previous frame difference, print it
        if k > 1 && abs(histDiff - histDiffs(k-1)) > spikeDifference
            fprintf('Major spike detected  between frame %d and frame %d with difference value: %d\n', k, k+1, histDiff);
        end
    end
end

% Plot histogram differences to visualize spikes at shot boundaries
figure;
plot(histDiffs, 'b-', 'LineWidth', 1.5); % Blue line for histogram differences
title('Histogram Difference Between Consecutive Frames');
xlabel('Frame Number');
ylabel('Histogram Difference');
grid on;
hold on;

% Highlight detected major spikes with red markers and display difference values
for i = 2:length(histDiffs)
    % Only plot if there's a major spike
    if abs(histDiffs(i) - histDiffs(i-1)) > spikeDifference
        plot(i, histDiffs(i), 'ro'); % Red dot for major spike
        text(i, histDiffs(i), num2str(histDiffs(i)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'Color', 'red'); % Display difference value
    end
end
hold off;

% Display total number of major spikes detected
totalShotBoundaries = sum(abs(diff(histDiffs)) > spikeDifference);
fprintf('Total number of major shot boundaries detected : %d\n', totalShotBoundaries);
