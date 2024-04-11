{
  description = "This is the runner for Project-W, a service that converts uploaded audio files into downloadable text transcripts using OpenAIs whisper AI model, hosted on a backend server and dedicated runners.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
  };

  outputs = inputs@{ nixpkgs, systems, ...}: 
  let
    pythonOverlay = import ./nix/overlay.nix;
    eachSystem = nixpkgs.lib.genAttrs (import systems);
    pkgsFor = eachSystem (system: 
      import nixpkgs { 
        inherit system; 
        #overlays add all new packages and their dependencies
        overlays = [pythonOverlay];
      }
    );
  in {
    packages = eachSystem (system: rec {
      default = project-W-runner;
      project-W-runner = pkgsFor.${system}.python3Packages.project-W-runner;
    });
    devShells = eachSystem (system: {
      default = import ./nix/shell.nix { pkgs=pkgsFor.${system}; };
    });
    nixosModules.default = import ./nix/module.nix inputs;
    overlays.default = pythonOverlay;
  };
}
