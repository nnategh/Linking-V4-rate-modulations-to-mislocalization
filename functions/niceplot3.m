function [line, y, sem] = niceplot3(xinput, yinput, win, c1, c2, c3)
%   Plot trial-averaged data with a shaded SEM band
%
%   [y, sem] = niceplot3(xinput, yinput, win, c1, c2, c3)
%
%   Inputs:
%     xinput      – 1 x T time axis vector
%     yinput      – N x T matrix (trials x time)
%     win         – smoothing window width (samples); use 1 for no smoothing
%     c1, c2, c3  – RGB colour components in [0, 1]
%
%   Outputs:
%     y   – 1 x T trial mean of smoothed data
%     sem – 1 x T standard error of the mean

smoothed = zeros(size(yinput));
for k = 1:size(yinput, 1)
    smoothed(k, :) = smoothdata(yinput(k, :), 'gaussian', win);
end

y   = mean(smoothed, 1, 'omitnan');
sem = std(smoothed, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(smoothed), 1));

% Build and draw shaded ribbon + mean line
x_patch = [xinput,       flip(xinput)];
y_patch = [(y + sem),    flip(y - sem)];

valid = ~isnan(y_patch);
fill(x_patch(valid), y_patch(valid), [c1 c2 c3], ...
     'LineStyle', 'none', 'FaceAlpha', 0.2);
hold on;
line = plot(xinput, y, 'Color', [c1 c2 c3], 'LineWidth', 2);
