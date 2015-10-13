%Function for data normalization:
%input:
%data = NxD data matrix (D=data dimension)
%mode = type of normalization.
%       1 = min-max normalization, where for each variable x:
%  norm_x = (x-min_x)/(max_x-min_x)*(max_range_x-min_range_x)+min_range_x,
%       in which min_range_x and max_range_x are the new max and min
%       value for the variable x
%       2 = standardization/Z-score, where for each variable x
%  norm_x = (x-mean_x)/sigma_x,
%       in which sigma_x is the stddev of x and mean_x is the mean of x
%data_range = range of new data in a row vector:
%   [min_x1 min_x2 ... min_xD ; max_x1 max_x2 ... max_xD]

function norm_x = My_normalize_data(data,mode,data_range)

[N D] = size(data);
norm_x = zeros([N D]);
switch mode
   case 1
      min_x = min(data);
      max_x = max(data);
      for j=1:D
      norm_x(:,j) = (data(:,j)-min_x(j)*ones(N,1))/(max_x(j)-min_x(j))*...
            (data_range(2,j)-data_range(1,j)) + data_range(1,j)*ones(N,1);
      end
   case 2
      mean_x = mean(data);
      sigma_x = std(data);
      for j=1:D 
      norm_x(:,j) = (data(:,j)-mean_x(j)*ones(N,1))/sigma_x(j);     
      end
end
