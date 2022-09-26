with open("gam.par", "w") as f:
    f.write("                         Parameters for GAM                                  \n")
    f.write("                         *******************                                 \n")
    f.write("                                                                             \n")
    f.write("START OF PARAMETERS:                                                         \n")
    f.write("gam.dat                                 -file with data                      \n")
    f.write("25 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 -number of var.,col numbers\n")
    f.write("-1.0e21     1.0e21                      -trimming limits                     \n")
    f.write("gam_out.out                             -file for variogram output           \n")
    f.write("1                                       -grid or realization number          \n")
    f.write("335 0.5 1                               -nx, xmn, xsiz                       \n")
    f.write("335 0.5 1                               -ny, ymn, ysiz                       \n")
    f.write("1 0 0                                   -nz, zmn, zsiz                       \n")
    f.write("1 100                                   -number of directions, number of lags\n")
    f.write("1  0  0                                 -ixd(1),iyd(1),izd(1)                \n")
    f.write("0                                       -standardize sill? (0=no, 1=yes)     \n")
    f.write("325                                     -number of variograms                \n")
    for v1 in range(1, 26):
        for v2 in range(v1, 26):
            f.write(str(v1) + " " + str(v2) + " 2      -tail, head, variogram type  \n")

