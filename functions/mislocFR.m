function [highFR, lowFR] = mislocFR(monkey, time, mode) 
    %{
    monkey: 1 == O, 2 == T;
    time: 1 == fixation, 2 == perisaccadic;
    mode: 1 == all probes, 2 == select probes
    %}

    % --- 1. Configuration & Setup ---
    base_dir = 'D:\';
    monkeys = {'O', 'T'};
    m_name = monkeys{monkey};
    
    if monkey == 1
        t_bin = [-250 -180 -70 0]; 
    elseif monkey == 2
        t_bin = [-250 -180 -150 -80]; 
    else
        t_bin = [-250 -150 -70 10];
    end
    
    % --- 2. Load Metadata ---
    load(fullfile(base_dir, sprintf('saved_variables\\%s_misloc_median.mat', m_name)), 'probes');
    select = squeeze(probes(:,1,:) > 3 & probes(:,1,:) < 9 & probes(:,2,:) > -10 & probes(:,1,:) < 5);
    
    T = readtable(fullfile(base_dir, sprintf('Recordings\\%s_sessions.xlsx', m_name)));
    sessions = table2cell(T(table2array(T(:,5)) > 0, :));
    sessions(:,1) = cellfun(@(x) double(string(x)), sessions(:,1), 'UniformOutput', false);
    
    select_sessions = sessions(sum(select,2) > 0 & cell2mat(sessions(:,5)) == 2, :);
    
    R = readtable(fullfile(base_dir, sprintf('Recordings\\%s_neurons.xlsx', m_name)));
    list = table2cell(R(R{:,3} > 0 & R{:,3} < 10 & R{:,10} ~= 1 & R{:,13} == 1, :));
    
    % Filter based on mode
    if mode == 1
        select_list = list;
    elseif mode == 2
        % Fast vectorized lookup instead of nested loops
        valid_session_ids = cell2mat(select_sessions(:,1));
        list_session_ids = cell2mat(list(:,1));
        select_list = list(ismember(list_session_ids, valid_session_ids), :);
        
        select = select(sum(select,2) > 0 & cell2mat(sessions(:,5)) == 2, :);
    end

    % --- 3. Process Units ---
    num_units = size(select_list, 1);
    highFR_all = cell(num_units, 9);
    lowFR_all = cell(num_units, 9);
    percentile = 0.45;
    
    for u = 1:num_units
        % Format date (automatically pads with 0 if 5 digits)
        date_str = sprintf('%06d', select_list{u,1}); 
        ch = select_list{u,2};
        unit = select_list{u,3};
        
        % File Paths
        psth_path = fullfile(base_dir, sprintf('saved_variables\\psth\\psth_%s_%d_%d.mat', date_str, ch, unit));
        vis_path = fullfile(base_dir, sprintf('saved_variables\\vis\\vis_%s.mat', date_str));
        
        load(psth_path, 'SpikeProbe');
        load(vis_path, 'Probe_OnsetfromSaccade', 'Conditions', 'xdva_probe', 'ydva_probe', 'cal_val');
        
        % Clean Spike data
        zero_rows = all(SpikeProbe == 0, 2);
        SpikeProbe(zero_rows, :) = nan;
        
        % Time group definition
        group = [Probe_OnsetfromSaccade > t_bin(1) & Probe_OnsetfromSaccade <= t_bin(2), ... 
                 Probe_OnsetfromSaccade > t_bin(3) & Probe_OnsetfromSaccade <= t_bin(4)];
             
        if mode == 2
            select_probes = select(cell2mat(select_sessions(:,1)) == select_list{u,1}, :);
        end
        
        % Apply Calibration 
        xdva_probeC = xdva_probe;
        ydva_probeC = ydva_probe;
        for c = 1:9
            idx = (Conditions == 50 + c);
            xdva_probeC(idx) = xdva_probe(idx) - cal_val(c, 1);
            ydva_probeC(idx) = ydva_probe(idx) - cal_val(c, 2);
        end
        saccade_land = [xdva_probeC' ydva_probeC'];

        % Process Probes
        for p = 1:9
            % Skip if mode 2 and probe not selected
            if mode == 2 && ~select_probes(p)
                continue;
            end
            
            % Condition and time filters
            cond_t = (Conditions == 50 + p) & group(:, time);
            cond_base = (Conditions == 50 + p) & group(:, 1);
            
            spike_after_p = SpikeProbe(cond_t, :);
            
            % Horizontal absolute mislocalization
            saccade_t_x = saccade_land(cond_t, 1);
            saccade_base_x = saccade_land(cond_base, 1);
            misloc_p = abs(saccade_t_x - median(saccade_base_x, 'omitnan'));
            
            % Sort and extract top/bottom percentiles
            [~, sorted_indices] = sort(misloc_p);
            cutoff = ceil(size(misloc_p, 1) * percentile);
            
            if cutoff > 0 
                bot_in = sorted_indices(1:cutoff);
                top_in = sorted_indices(end-cutoff+1:end);
                
                highFR_all{u, p} = smoothdata(spike_after_p(top_in, :), 'gaussian', 20);
                lowFR_all{u, p} = smoothdata(spike_after_p(bot_in, :), 'gaussian', 20);
            end
        end
    end

    % --- 4. Concatenate Across Probes ---
    highFR = cell(num_units, 1);
    lowFR = cell(num_units, 1);
    
    for n = 1:num_units
        highFR{n} = vertcat(highFR_all{n, :});
        lowFR{n} = vertcat(lowFR_all{n, :});
    end
end