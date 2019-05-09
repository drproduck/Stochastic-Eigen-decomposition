% Add folders to path.

addpath(pwd);

cd algorithm/;
addpath(genpath(pwd));
cd ..;

cd utils/;
addpath(genpath(pwd));
cd ..;

cd solvers/;
addpath(genpath(pwd));
cd ..;

cd manopt;
addpath(genpath(pwd));
cd ..;

cd tools/;
addpath(genpath(pwd));
cd ..;

setenv('EDITOR', 'vim')
