% videoShotDetectionScript.m
% Script to run video shot detection on an input video file and display matrices and shot boundaries.

% Define the video file path
videoPath = '/MATLAB Drive/11.mp4';  % Replace with the actual video path

% Set matrix size and threshold for shot detection
matrixSize = [5, 5];   % Size of the downscaled matrix
highThreshold = 100;   % Set a higher threshold for significant shot boundaries

% Call the videoShotDetection function
[frameDifferences, shotBoundaryCount] = videoShotDetection(videoPath, matrixSize, highThreshold);

% Display the results
disp('Frame Differences:');
disp(frameDifferences);
fprintf('Total number of shot boundaries detected: %d\n', shotBoundaryCount);

% ---------------------------------------------------
% videoShotDetection.m
% Function to perform shot detection based on large frame differences

function [frameDifferences, shotBoundaryCount] = videoShotDetection(videoPath, matrixSize, highThreshold)
    % videoShotDetection: Detect shot changes in a video based on frame differences.
    % Parameters:
    %   videoPath     - Path to the video file (e.g., 'your_video.mp4')
    %   matrixSize    - Size of the resized frame matrix (default: [5, 5])
    %   highThreshold - Threshold for detecting significant shot changes
    %
    % Output:
    %   frameDifferences   - Array of differences between consecutive frames
    %   shotBoundaryCount  - Total number of detected shot boundaries

    % Read the video
    vidObj = VideoReader(videoPath);
    
    % Initialize variables
    previousFrameMatrix = [];
    frameDifferences = [];
    frameCount = 0;
    shotBoundaryCount = 0;  % Initialize shot boundary counter

    % Loop through frames
    while hasFrame(vidObj)
        % Read the current frame and convert it to grayscale
        frame = readFrame(vidObj);
        grayFrame = rgb2gray(frame);
        
        % Resize frame to the specified matrix size (e.g., 5x5)
        resizedFrame = imresize(grayFrame, matrixSize);
        
        % Convert to double for numerical operations
        frameMatrix = double(resizedFrame);

        % Print the 5x5 matrix for the current frame
        fprintf('Frame %d matrix:\n', frameCount);
        disp(frameMatrix);

        % If this is not the first frame, calculate the difference
        if ~isempty(previousFrameMatrix)
            % Calculate the absolute difference between consecutive frames
            difference = abs(frameMatrix - previousFrameMatrix);
            frameDifference = sum(difference(:));

            % Append the frame difference to the result
            frameDifferences = [frameDifferences, frameDifference];

            % Check if the difference exceeds the high threshold
            if frameDifference > highThreshold
                fprintf('Shot boundary detected at frame %d with difference %f\n', frameCount, frameDifference);
                shotBoundaryCount = shotBoundaryCount + 1;  % Increment shot boundary counter
            end
        end

        % Update previous frame
        previousFrameMatrix = frameMatrix;
        frameCount = frameCount + 1;
    end

    % Release the video object (good practice)
    clear vidObj;

    % Display total number of shot boundaries
    fprintf('Total number of shot boundaries detected: %d\n', shotBoundaryCount);
end
