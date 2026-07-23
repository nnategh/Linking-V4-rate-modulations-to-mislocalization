%% Compute Center of Mass for High vs Low during Fixation and Perisaccadic

% Load probe coordinates topology (3x3 grid layout)
grid_map = [6 5 4; 
            7 1 3; 
            8 9 2];

% Preallocate output matrices
num_units = size(list, 1);
ST  = nan(num_units, 2);
RF  = nan(num_units, 2);
RF_ST = nan(num_units, 3);

fix_high_center    = nan(num_units, 2);
fix_low_center     = nan(num_units, 2);
peri_high_center   = nan(num_units, 2);
peri_low_center    = nan(num_units, 2);

fix_high_center_ST  = nan(num_units, 3);
fix_low_center_ST   = nan(num_units, 3);
peri_high_center_ST = nan(num_units, 3);
peri_low_center_ST  = nan(num_units, 3);

% Preallocate 3x3 matrices used per unit iteration
gridx = nan(3, 3);
gridy = nan(3, 3);
fix_high_matrix  = nan(3, 3);
fix_low_matrix   = nan(3, 3);
peri_high_matrix = nan(3, 3);
peri_low_matrix  = nan(3, 3);

for n = 1:num_units
    date_str = sprintf('%06d', list{n, 1});
    
    % Load visual probe data
    vis_file = fullfile('D:\saved_variables\vis', ['vis_' date_str '.mat']);
    load(vis_file, 'probex', 'probey');
    
    % Saccade target vector & direction correction
    ST(n, :) = str2num(list{n, 9}); %#ok<ST2NM>
    if ST(n, 1) < 0
        probex  = -probex;
        ST(n, 1) = -ST(n, 1);
    end
    
    % Receptive Field & distance to Saccade Target
    RF(n, :)  = [probex(max_pq(n)), probey(max_pq(n))];
    RF_ST(n, :) = calc_rel_dist(RF(n, :), ST(n, :));
    
    % Map probe positions and response means into 3x3 matrices
    gridx(grid_map) = probex;
    gridy(grid_map) = probey;
    
    fix_high_matrix(grid_map)  = fix_high_mean(n, :);
    fix_low_matrix(grid_map)   = fix_low_mean(n, :);
    peri_high_matrix(grid_map) = peri_high_mean(n, :);
    peri_low_matrix(grid_map)  = peri_low_mean(n, :);
        
    % Compute Center of Mass
    [fix_high_center(n,1),  fix_high_center(n,2)]  = center_of_mass(fix_high_matrix,  gridx, gridy);
    [fix_low_center(n,1),   fix_low_center(n,2)]   = center_of_mass(fix_low_matrix,   gridx, gridy);
    [peri_high_center(n,1), peri_high_center(n,2)] = center_of_mass(peri_high_matrix, gridx, gridy);
    [peri_low_center(n,1),  peri_low_center(n,2)]  = center_of_mass(peri_low_matrix,  gridx, gridy);
    
    % Center of Mass distances relative to Saccade Target
    fix_high_center_ST(n, :)  = calc_rel_dist(fix_high_center(n, :),  ST(n, :));
    fix_low_center_ST(n, :)   = calc_rel_dist(fix_low_center(n, :),   ST(n, :));
    peri_high_center_ST(n, :) = calc_rel_dist(peri_high_center(n, :), ST(n, :));
    peri_low_center_ST(n, :)  = calc_rel_dist(peri_low_center(n, :),  ST(n, :));

    fprintf('Processed unit %d/%d\n', n, num_units);
end

% --- Local Helper Function ---
function dist_vec = calc_rel_dist(pt, target)
    % Returns [abs(dx), abs(dy), euclidean_distance]
    diff_vec = pt - target;
    dist_vec = [abs(diff_vec(1)), abs(diff_vec(2)), norm(diff_vec)];
end