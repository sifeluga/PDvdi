# PDvdi
Poisson disc sampling with variable density and temporal incoherence

Note: this python implementation is a C++ port for convience purposes, where runtimes are around a 100-fold longer. 
(In C++, the generation of an exemplary pattern executes in typically in less than 50 ms)

    Example usage: 'PDvdiSampler.py 150 110 12.5 -N 2 -o 1'
    generates two patterns of size 150x110 and 12.5-fold undersampling with a graphical output

    W           Pattern width
    H           Pattern height
    AF          Acceleration factor (AF) > 1.0

    optional arguments:
    -h, --help  show this help message and exit
    -N       Number of patterns to generate jointly
    -NN      Number of neighbors (NN) ~[20;80]
    -p       Variable density: Power to raise of distance from center ~[1.5;2.5]
    -i       Temporal incoherence [0;1[
    -o       Output: 0 - graphical, 1: textfile of sample locations, 2: both
