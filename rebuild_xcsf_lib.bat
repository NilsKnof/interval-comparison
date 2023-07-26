cd ..\xcsf\build
cmake -DCMAKE_BUILD_TYPE=Release -DXCSF_PYLIB=ON -DENABLE_TESTS=ON -G "MinGW Makefiles" ..
cmake --build . --config Release
rmdir /s /q "D:\Coding Stuff\Bachelorarbeit\repos\interval-comparison\libs\xcsf"
xcopy /s /i "D:\Coding Stuff\Bachelorarbeit\repos\xcsf\build\xcsf" "D:\Coding Stuff\Bachelorarbeit\repos\interval-comparison\libs\xcsf"
pause 
