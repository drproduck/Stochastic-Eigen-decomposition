function scatter2d(fea, gnd)

if exist('gnd', 'var')
	scatter(fea(:,1),fea(:,2),5,gnd)
else
	scatter(fea(:,1),fea(:,2),5)
end
