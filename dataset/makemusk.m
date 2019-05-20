clear;
f = fopen('musk.data', 'r');
str = [repmat('%s',1,2), repmat('%f', 1, 167)];
data = textscan(f, str, 'Delimiter', ',');
fea = zeros(6598, 166);
gnd = data{169};
for i = 3:168
    fea(:, i-2) = data{i};
end

% fea = zeros(67557, 42);
% for i = 1:42
%     x = data{i};
%     y = zeros(67557, 1);
%     for j = 1:67557
%         if x{j} == 'x'
%             y(j) = 1;
%         elseif x{j} == 'o'
%             y(j) = 2;
%         elseif x{j} == 'b'
%             y(j) = 3;
%         else
%             error('no such element');
%         end
%     end
%     fea(:,i) = y;
% end