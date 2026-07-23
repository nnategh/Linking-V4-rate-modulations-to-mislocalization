function [x, y] = center_of_mass(matrix, gridx, gridy)

[rows, cols] = size(matrix);

if ~exist('gridx','var')
  [gridx, gridy] = meshgrid(1:cols, 1:rows);
end

total_mass = nansum(matrix(:));
x = nansum(gridx(:) .* matrix(:)) / total_mass;
y = nansum(gridy(:) .* matrix(:)) / total_mass;

end