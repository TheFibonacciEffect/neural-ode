Here is some feedback regarding this code, please incorperate it
use jj as version control, dont abandon any revisions, you may edit existing revisions
1. the random number generator is not a parameter of interrest, but it should still be fixed, move it to training_framework.jl
2. the current training loss should only be printed 10 times in total, so calculate what the integer N is sp that it is only prined each Nth iteration, also print the current itteration
3. the models and plots should be saved subfolders, also move the models and plots there
4. the code should be in an src folder
5. test the code by running it
