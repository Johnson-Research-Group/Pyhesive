// Gmsh project created on Tue Dec  1 17:35:28 2020
SetFactory("OpenCASCADE");
//+
Box(1) = {-1, -1, 0, 1, 1, 1};
//+
Box(2) = {-1, -1, 0, 1, 1, 1};
//+
Characteristic Length {3, 4, 8, 7, 1, 5, 6, 2} = 0.48;
