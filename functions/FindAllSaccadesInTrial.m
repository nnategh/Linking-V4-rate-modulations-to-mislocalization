function [sac, micro] = FindAllSaccadesInTrial(eye_pos, graphics, noise_elim)

% function FindAllSaccadesInTrial
%
% Uses a conjunction of a displacement test between moving boxcars, a
% significance test on the displacement, and a velocity threshold to mark
% saccades and microsaccades in eye data
%
% Input arguments:
%       eye_pos: a 2-column matrix of eye positions (x and y) sampled at
%           1kHz and in units of degrees
%       graphics (optional): 1 to show eye traces, microsaccades and
%           saccades, 0 to suppress [0 is default]
%       noise_elim (optional): 1 to use Chronux toolbox to eliminate line
%           noise around 20Hz, 60Hz and 153Hz, 0 to skip [1 is default]
%
% Outputs:
%       sac: a structure that includes the times, start/end positions, and
%          vectors of all saccades, along with traces of the eye data
%       micro: same as 'sac' except for the microsaccades

if nargin == 1
    graphics = 0; noise_elim = 1;
elseif nargin == 2
    noise_elim = 1;
end

% Parameters
box = 10; % ms long boxcar, default = 50
gap = 1; % ms between boxcars, default = 5
v_thresh = 50; % vel thresh in degrees per second, default = 150
v_sample_thresh = 1; % vel must stay above thresh for at least this many ms, default = 3
d_thresh = 0.2; % displacement thresh in degrees, default = 6
min_between = 20; % can't have two sacs/micros within this many ms, default = 20
micro_cutoff = 2; % degrees.  Under this is a micro, over is a regular sac, default = 7


eye_pos(:,1) = gauss_smooth(eye_pos(:,1),5);
eye_pos(:,2) = gauss_smooth(eye_pos(:,2),5);
n_micro = 0;
n_reg = 0;
micro = struct([]);
sac = struct([]);

n = length(eye_pos);

if noise_elim
    % There's sometimes a lot of noise at around 20Hz and around 153Hz in
    % Buddha's eye coil.  Remove these lines here if they're significant.
    p = eye_pos(~isnan(eye_pos(:,1)),:);
    if isempty(p); return; end
    x = p(:,1);     y = p(:,2);     params.Fs = 1000;
    [datafit,amps,freqs] = fitlinesc(x,[],params,0.05,'n');
    noise_bands = freqs{1}((freqs{1} > 19 & freqs{1} < 21) | (freqs{1} > 59 & freqs{1} < 61) | (freqs{1} > 151 & freqs{1} < 155));
    for i = 1:length(noise_bands)
        elim_line = fitlinesc(x,noise_bands(i),params,[],'n');
        x = x - elim_line;
    end
    [datafit,amps,freqs] = fitlinesc(y,[],params,0.05,'n');
    noise_bands = freqs{1}((freqs{1} > 19 & freqs{1} < 21) | (freqs{1} > 59 & freqs{1} < 61) | (freqs{1} > 151 & freqs{1} < 155));
    for i = 1:length(noise_bands)
        elim_line = fitlinesc(y,noise_bands(i),params,[],'n');
        y = y - elim_line;
    end
    eye_pos(~isnan(eye_pos(:,1)),:) = [x y];
end

% Mark times where the displacement between boxcars exceeds the threshold
x_jiggle = zeros(box,length(eye_pos)+box); % use this as fast way to compute running medians
y_jiggle = zeros(box,length(eye_pos)+box);
for i = 1:box
    x_jiggle(i,i:i+length(eye_pos)-1) = eye_pos(:,1)';
    y_jiggle(i,i:i+length(eye_pos)-1) = eye_pos(:,2)';
end
xm = median(x_jiggle);  xm = xm(round(box/2)+1:end-(box/2));
ym = median(y_jiggle);  ym = ym(round(box/2)+1:end-(box/2));
h_d = false(1,n); % now take the medians and compare at center of boxcars
for i = box+gap:n-box-gap
% [[xm(i-round(gap/2)-round(box/2)),ym(i-round(gap/2)-round(box/2))]; [xm(i+round(gap/2)+round(box/2)),ym(i+round(gap/2)+round(box/2))]]
    h_d(i) = pdist([[xm(i-round(gap/2)-round(box/2)),ym(i-round(gap/2)-round(box/2))]; [xm(i+round(gap/2)+round(box/2)),ym(i+round(gap/2)+round(box/2))]]) > d_thresh;
%     
%     h_d(i) = Distance([xm(i-round(gap/2)-round(box/2)),ym(i-round(gap/2)-round(box/2))], ...
%         [xm(i+round(gap/2)+round(box/2)),ym(i+round(gap/2)+round(box/2))]) > d_thresh;
end

% Check where displacement exceeds thresh and discard those times if they
% aren't significant in either the x or y.  Should eliminate places marked due to noise
sig_spots = find(h_d==1);
for i = 1:length(sig_spots)
    h_d(sig_spots(i)) = (kstest2(eye_pos(sig_spots(i)-box-round(gap/2)+1:sig_spots(i)-round(gap/2),1),...
        eye_pos(sig_spots(i)+round(gap/2)+1:sig_spots(i)+box+round(gap/2),1),0.01,'unequal')) | ...
        (kstest2(eye_pos(sig_spots(i)-box-round(gap/2)+1:sig_spots(i)-round(gap/2),2),...
        eye_pos(sig_spots(i)+round(gap/2)+1:sig_spots(i)+box+round(gap/2),2),0.01,'unequal'));
end

% Mark times where the velocity is above the threshold for several samples
% in a row
eye_vel = [0; (diff(eye_pos(:,1)).^2 + diff(eye_pos(:,2)).^2).^(0.5)];
h_v = conv(single(eye_vel >= v_thresh/1000),ones(v_sample_thresh,1))==v_sample_thresh;
h_v = h_v(v_sample_thresh:end)';

% Find intersections, narrow down to single time stamp at center of the
% continuous interval
s_times = find(h_d & h_v);
if isempty(s_times); return; end
streak_starts = s_times(1);
if length(s_times)>1 && s_times(2)-s_times(1) ~= 1; streak_ends = s_times(1);
else streak_ends = [];
end
for ptr = 2:length(s_times)-1
    if s_times(ptr) ~= s_times(ptr-1)+1
        streak_starts(end+1) = s_times(ptr);
    end
    if s_times(ptr) ~= s_times(ptr+1)-1
        streak_ends(end+1) = s_times(ptr);
    end
end
if length(streak_ends)==length(streak_starts)-1
    streak_ends(end+1) = s_times(end);
end
mids = round(mean([streak_starts; streak_ends])); % midpoints of all of the intervals that are sacs/microsacs
% Get rid of streaks that follow another streak too quickly
good_sacs = mids(1);
for i = 2:length(mids)
    if mids(i)-mids(i-1) > min_between
        good_sacs(end+1) = mids(i);
    end
end

% Move these initial guesses to the highest-vel time in close proximity
for i = 1:length(good_sacs)
    snippet = eye_vel(good_sacs(i)-round(box/2) : good_sacs(i)+round(box/2));
    good_sacs(i) = good_sacs(i) + find(snippet == max(snippet),1,'first') - round(box/2);
end

% Look at amplitude of the eye movements to determine sac vs. micro, and
% for the saccades go forward and backward in time to get first and final
% past-threshold times.
for i = 1:length(good_sacs)

    % Find where the eye movement starts and stops by going
    % forward/backward until the eye vel is first/last above the thresh
    j = 1;
    while (good_sacs(i)+j)<length(eye_vel) && eye_vel(good_sacs(i)+j)>v_thresh/1000
        j = j+1;
    end
    sac_stop_idx = good_sacs(i)+j;

    j = -1;
    while (good_sacs(i)+j)<length(eye_vel) && eye_vel(good_sacs(i)+j)>v_thresh/1000
        j = j-1;
    end
    sac_start_idx = good_sacs(i)+j;

    % Now figure out the start and ending positions and determine if it was
    % a micro or normal
    start_pos = [xm(max(1,sac_start_idx - round(box/2))), ym(max(1,sac_start_idx - round(box/2)))];
    stop_pos = [xm(min(length(xm),sac_stop_idx + round(box/2))), ym(min(length(xm),sac_stop_idx + round(box/2)))];

    if pdist([start_pos;stop_pos]) > micro_cutoff
        % Regular saccade
        n_reg = n_reg + 1;
        sac(n_reg).start_pos = start_pos;
        sac(n_reg).stop_pos = stop_pos;
        sac(n_reg).start_time = sac_start_idx;
        sac(n_reg).stop_time = sac_stop_idx;
        sac(n_reg).pre_sac_100ms = eye_pos(max(1,sac_start_idx-100):sac_start_idx-1,:);
        sac(n_reg).trace = eye_pos(sac_start_idx:sac_stop_idx,:);
        sac(n_reg).post_sac_100ms = eye_pos(sac_stop_idx+1:min(end,sac_stop_idx+100),:);
    else        % microsaccade
        n_micro = n_micro + 1;
        micro(n_micro).time = sac_start_idx;
        micro(n_micro).start_pos = start_pos;
        micro(n_micro).stop_pos = stop_pos;
        micro(n_micro).start_time = sac_start_idx;
        micro(n_micro).stop_time = sac_stop_idx;
    end
end

% Add in polar coordinates for the saccades and microsaccades
for i = 1:length(sac)
    if ~isfield(sac, 'stop_pos'); continue; end
    cart_vec = sac(i).stop_pos - sac(i).start_pos;
    [sac(i).theta, sac(i).r] = cart2pol(cart_vec(1), cart_vec(2));
end
for i = 1:length(micro)
    if ~isfield(micro, 'stop_pos'); continue; end
    cart_vec = micro(i).stop_pos - micro(i).start_pos;
    [micro(i).theta, micro(i).r] = cart2pol(cart_vec(1), cart_vec(2));
end

if graphics
    % Optional plotting commands
%     figure; 
    hold on;
    plot(1:length(eye_pos),eye_pos(:,1),'b');
    plot(1:length(eye_pos),eye_pos(:,2),'b');
    for i = 1:length(sac)
        plot(sac(i).start_time,0,'go');
        plot(sac(i).stop_time,0,'ro');
    end
    for i = 1:length(micro)
        plot(micro(i).time,0,'b*');
    end
end

return;

