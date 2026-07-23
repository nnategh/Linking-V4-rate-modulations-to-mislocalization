%% Calculate mislocalization and localization error for all sessions
clear;

% --- Configuration & Constants ---
direction = 50;
C = direction + [6 5 4 7 1 3 8 9 2];
t_bin = [-250 -180 -70 -0]; 
window = 50; % ms

% --- Load Metadata ---
T = readtable('D:\Recordings\sessions.xlsx');
list = table2cell(T(table2array(T(:,5)) > 0, :)); 

num_sessions = length(list);

% --- Preallocation ---
magn_x  = nan(num_sessions, 9, 800);
magn_px = nan(num_sessions, 9, 800);
magn_y  = nan(num_sessions, 9, 800);
magn_py = nan(num_sessions, 9, 800);
magn    = nan(num_sessions, 9, 800);
magn_p  = nan(num_sessions, 9, 800);
probes  = nan(num_sessions, 2, 9);

% --- Main Session Loop ---
for n = 1:num_sessions
    % Determine directional multiplier
    d = iff(list{n,5} == 1, -1, 1); 
    
    % Format date string cleanly
    date_str = num2str(list{n,1}, '%06d');
    
    % File paths
    vis_path = fullfile('D:\saved_variables\vis', ['vis_' date_str '.mat']);
    hist_path = fullfile('D:\saved_variables\histograms', ['histograms_' date_str '.mat']);
    
    % Load session variables
    load(vis_path, 'probex', 'probey', 'cal_val', 'Conditions', 'xdva_probe', 'ydva_probe', 'Probe_OnsetfromSaccade');
    load(hist_path, 'reaction_time');
    
    % Data filtering
    Probe_OnsetfromSaccade(Probe_OnsetfromSaccade > 400) = nan;
    probes(n,:,:) = [probex; probey];
    
    % --- Apply Calibration across all trials ---
    xdva_probeC = xdva_probe;
    ydva_probeC = ydva_probe;
    for c = 1:9
        cond_idx = (Conditions == (direction + c));
        xdva_probeC(cond_idx) = xdva_probe(cond_idx) - cal_val(c, 1);
        ydva_probeC(cond_idx) = ydva_probe(cond_idx) - cal_val(c, 2);
    end
    saccade_land = [xdva_probeC', ydva_probeC'];
    
    % Baseline grouping based on time bins
    group = [Probe_OnsetfromSaccade > t_bin(1) & Probe_OnsetfromSaccade <= t_bin(2), ...
             Probe_OnsetfromSaccade > t_bin(3) & Probe_OnsetfromSaccade <= t_bin(4)];
    
    % Calculate landing locations baseline for the session
    landing_locations_m = nan(9, 2, size(group, 2));
    for c = 1:9
        for g = 1:size(group,2)
            idx = (Conditions == (direction + c)) & group(:, g);
            if any(idx)
                landing_locations_m(c, :, g) = [nanmedian(saccade_land(idx, 1)), nanmedian(saccade_land(idx, 2))];
            end
        end
    end
    
    % Sliding Window Time Analysis
    mint = min(Probe_OnsetfromSaccade);
    maxt = max(Probe_OnsetfromSaccade);
    if isempty(mint) || isempty(maxt); continue; end % Skip if no valid data
    
    time = mint:maxt;
    half_window = window / 2;
    
    for p = 1:9  
        cond_c = C(p) - direction;
        
        % Extract baseline coordinates for this condition
        base_x = landing_locations_m(cond_c, 1, 1);
        base_y = landing_locations_m(cond_c, 2, 1);
        
        % Loop through valid time windows
        for t = 26:(length(time) - half_window)
            tgroup = Probe_OnsetfromSaccade >= (time(t) - half_window) & ...
                     Probe_OnsetfromSaccade <= (time(t) + half_window);
                 
            landing_locationt = saccade_land(Conditions == C(p) & tgroup, :);
            
            if ~isempty(landing_locationt)
                med_t_x = nanmedian(landing_locationt(:, 1));
                med_t_y = nanmedian(landing_locationt(:, 2));
                mean_t_y = nanmean(landing_locationt(:, 2));
                
                time_idx = time(t) + 401;
                
                % Store calculations across full session      
                magn_x(n, p, time_idx)  = d * (med_t_x - base_x);
                magn_px(n, p, time_idx) = d * (-probex(cond_c) + med_t_x);
                magn_y(n, p, time_idx)  = med_t_y - base_y;
                magn_py(n, p, time_idx) = -probey(cond_c) + mean_t_y; 
                magn(n, p, time_idx)    = sqrt((base_x - med_t_x)^2 + (base_y - med_t_y)^2);
                magn_p(n, p, time_idx)  = sqrt((probex(cond_c) - med_t_x)^2 + (probey(cond_c) - med_t_y)^2);
            end
        end
    end
    
    fprintf('%d/%d\n', n, num_sessions);
end

% --- Inline ternary operator helper function ---
function val = iff(condition, true_val, false_val)
    if condition; val = true_val; else; val = false_val; end
end