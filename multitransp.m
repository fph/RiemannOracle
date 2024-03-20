function B = multitransp(A, unused)
% this function is here to disable a compatibility check that slows down
% Manopt sensibly on smaller examples. 
% If you have Matlab R2020a or earlier, you can just remove this file and
% everything will still works.

B = pagetranspose(A);

