cd src/external/PyMiniSolvers
make

cd ../abc
make
make libabc.a
cp abc.rc ../../
cd ../aiger/aiger
# may need to change the complier in makefile to clang for macos
./configure.sh && make
cd ../cnf2aig
./configure && make

cd ../../..

