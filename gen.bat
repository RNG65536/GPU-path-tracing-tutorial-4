rmdir /s /q _build
mkdir _build
cd _build
cmake .. -G "Visual Studio 15 Win64" -DGLEW_ROOT_PATH="C:/Deps/glew" -DGLUT_ROOT_PATH="C:/Deps/freeglut"
cd ..