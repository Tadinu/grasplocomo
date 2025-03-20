mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../release
make -j8 && make install
