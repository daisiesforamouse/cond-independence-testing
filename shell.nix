{ pkgs, lib, stdenv, ... }:

let
  pythonPackages = pkgs.python3Packages;
in
pkgs.mkShell rec {
  buildInputs = [
    pythonPackages.python
    pythonPackages.venvShellHook
    pythonPackages.tkinter
    pkgs.zlib
  ];

  venvDir = "./.venv";

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
  '';
  
  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
  '';
}
