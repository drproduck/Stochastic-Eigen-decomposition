% % construct a interweaving half circles dataset
% centers = [2,2; 6,5];
% radii = [3,3];
% [x1, y1] = circle(centers(1,:), radii(1), 1);
% [x2, y2] = circle(centers(2,:), radii(2), 0);
% fea = cat(2, cat(1,x1,x2), cat(1,y1,y2));
% gnd = cat(1, ones(50, 1) * 1, ones(50, 1) * 2);
% save('circle.mat', 'fea', 'gnd');
% 
% 
% scatter(cat(1, x1, x2), cat(1, y1, y2))
% 
% function [x,y] = circle(center, radius, up)
% 
% bl = -radius + center(1);
% 
% x = bl + rand(50, 1) * (2 * radius);
% if up
%     y = sqrt(radius(1) ^ 2 - (x - center(1)) .^ 2) + center(2);
% else
%     y = -sqrt(radius(1) ^ 2 - (x - center(1)) .^ 2) + center(2);
% end
% end


%using generatedata to generate circle data

close all;
clear all;
clc;

nbclasses = 2;
nbsamples = 200;
dim = 2;

details = cell(nbclasses, 5);
details{1, 1} = [250 150]; %center for class 1
details{1, 2} = 250; %radius for class 1
details{1, 3} = [20 20]; %std for class 1
details{1, 4} = 'lhalf';
details{1, 5} = 100;

details{2, 1} = [0 0]; %center for class 2
details{2, 2} = 250; %radius for class 2
details{2, 3} = [20 20]; %std for class 2
details{2, 4} = 'uhalf';
details{2, 5} = 100;

%details{3, 1} = [0 0]; %mean for class 3
%details{3, 2} = 150; %radius for class 3
%details{3, 3} = [5 5]; %std for class 3


[fea, gnd] = generatedata(nbsamples, nbclasses, dim, 'circles', details, true);
save('twomoons_small.mat', 'fea', 'gnd');

function [data, class] = generatedata(nbsamples, nbclasses, dim, dist, details, blackplot)
%nbsamples is used to specify how many data samples to generate
%
%nbclasses is used to specify how many classes to consider when generating
%the data
%
%dim is the dimension of each sample (for graphical representation use dim
%equal to 2)
%
%dist specify how the data are generated:
%'gaussian' means the data will be a gaussian distribution with a certain
%mean and standard deviation. details should be a cell (nbclasses, 2) and
%each row contains first the mean and second the std, and each mean or std
%is a vector of dimension dim
%
%'circles' means the data will be distributed as circles with a certain
%mean which means the center, a radius and a std to express how much points
%will deviate from the radius, details should be a cell (nbclasses, 5) and
%each row contains first the mean of dimension dim, then the radius of
%dimension 1, and std of dimension dim. the fourth detail is:
%'complete' for a whole circle
%'uhalf' for the upper half of the circle
%'lhalf' for the lower half of the circle
%and the fifth detail is the wanted percentage of the circle
%
%details contains in each row informations about the type of distribution
%
%blackplot when set to positive integer means that all data will be plotted
%as black circles if dimension is two (used for unsupervised learning
%

data = randn(nbsamples, dim);
class = randi(nbclasses, [1, nbsamples]);

if(blackplot)

    plotchoice = {'ko','ko','ko','ko','ko'};
else
    
    plotchoice = {'bo','r+','md','k*','wv'};
end

if(strcmp(dist, 'gaussian') > 0)
    
    for i = 1: nbsamples
        
        data(i, :) = details{class(i), 1} + details{class(i), 2} .* data(i, :);
    end
    
    if(dim == 2)
        
        figure;
        for i = 1: nbclasses

            hold on;
            points = data(class == i, :);
%             plot(points(:,1), points(:,2), plotchoice{i});
            scatter(points(:,1), points(:,2),5);
        end
        title('generated data');
        grid on;
    end
end

if(strcmp(dist, 'circles') > 0)
    
    uhalf = ones(nbclasses, 1);
    lhalf = ones(nbclasses, 1);
    
    for i = 1: nbclasses
        
        if(strcmp(details{i, 4}, 'uhalf'))
            uhalf(i) = -1;
        end
        
        if(strcmp(details{i, 4}, 'lhalf'))
            lhalf(i) = -1;
        end
    end
    
    for i = 1: nbsamples
        
        edge = round(details{class(i), 2} * details{class(i), 5} / 100);
        coord = randi([-edge edge], [1, dim]);
        if(randi([-3 3] > 0))
            
            coord(end) = -1 * uhalf(class(i))* sqrt(details{class(i), 2} ^ 2 - (sum(coord(1: dim - 1) .^ 2)));
        else
            
            coord(end) = 1 * lhalf(class(i)) * sqrt(details{class(i), 2} ^ 2 - (sum(coord(1: dim - 1) .^ 2)));
        end
        
        data(i, :) = coord + details{class(i), 1} + details{class(i), 3} .* data(i, :);
    end
    
    if(dim == 2)
        
        figure;
        for i = 1: nbclasses

            hold on;
            points = data(class == i, :);
%             plot(points(:,1), points(:,2), plotchoice{i});
            scatter(points(:,1), points(:,2),5);

        end
        title('generated data');
        grid on;
    end
end
class = class';
end