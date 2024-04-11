let
  dontCheckPythonPkg = drv: drv.overridePythonAttrs (old: { doCheck = false; });
  myPythonPackages = ps: with ps; [
    #all required dependencies + this projects package itself (required for sphinx)
    (dontCheckPythonPkg project-W-runner)

    #optional dependencies: tests
    pytest
    pytest-cov
  ];
in
{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  buildInputs = with pkgs; [
    (python3.withPackages myPythonPackages)
  ];
}
