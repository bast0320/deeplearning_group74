Explainer for the files.

PaiNN implementation in one python file: painn_onefile_clean.py. Here we did not use any inspiration from the internet or source code. 

The SWAclean_hpc.py file was used to run the files on DTU’s HPC GPU clusters. For this we were inspired for some parts of the code by PyG and other implementations and the source code (to fix for example an exploding behaviour in the delta_v). See credits. 

All the clean files are the part of the PaiNN implementation	. 

test_xxx are test function which is not needed to run the actual code. 

We have also included submit bash scripts to be able to run the code on the HPC. If everything is kept as is the runtime is approx. 14 hours for 150 epochs. 

You should run the "run_me.py" file.

Group 74
Project 10

Credit: https://github.com/MaxH1996/PaiNN-in-PyG/tree/main
