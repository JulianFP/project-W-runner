(final: prev: {
  python3 = prev.python3.override {
    packageOverrides = pyfinal: pyprev: {
      pyaml-env = prev.callPackage ./pkgs/pyaml-env.nix { };
      project-W-runner = prev.callPackage ./pkgs/project-W-runner.nix { };
    };
  };
  python3Packages = final.python3.pkgs;
})
