mkdir -p build
pushd build

cmake .. -DCMAKE_INSTALL_PREFIX=../Release \
	 -DCMAKE_BUILD_TYPE:STRING=Release
cmake --build . -j8 --config=Release
cmake --install .
popd
