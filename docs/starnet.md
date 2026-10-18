# StarNet setup and recovery

StarNet2 is a separate executable used by Siril. It is not a Python package;
`uv sync` does not install it. Keep its executable, model package, and runtime
libraries together. Do not add its binaries or weights to this repository.

## Install and configure

1. Download the CLI package for your operating system and CPU from the
   [official StarNet downloads](https://starnetastro.com/cli-tools/starnet/).
   Read and accept the included license before use. Use the CLI package, not
   the PixInsight plugin.
2. Extract the complete package into a permanent directory. For Apple Silicon,
   use the ARM64 CoreML package and retain `StarNet2_weights.mlpackage` and `lib/`
   alongside `starnet2`.
3. Run the executable with `--version` from a terminal. Resolve any launch error
   before configuring Siril.
4. In Siril, open Preferences > Miscellaneous and set **StarNet Executable
   Location** to the executable itself. Leave the weights override empty to use
   the bundled model. Save the preferences and restart Siril.
5. In a fresh Siril session, run `get core.starnet_exe` to confirm the saved path.

See [Siril's StarNet documentation](https://siril.readthedocs.io/en/stable/processing/stars/starnet.html)
and [StarNet's package layout](https://starnetastro.com/documentation/cli-installers/).

## Verified local installation

The M42 machine uses Siril 1.4.3 on Apple Silicon. StarNet2 CLI 2.6.2-0242
(CoreML ARM64) is installed under:

```text
~/Applications/StarNet2/starnet2_macos-arm64_2.6.2-0242_COREML_arm64_cli/
```

The official ZIP is `starnet2_macos-arm64_2.6.2-0242_COREML_arm64_cli.zip`.
Its verified SHA-256 is:

```text
dfeb471e983a9de1233b439a8c56d3e373ed26e37c85aababe737187d164d20f
```

The executable's Developer ID signature verifies as Mikita Misiura,
team `K33XTRCS9T`. Check the installed executable with:

```bash
"$HOME/Applications/StarNet2/starnet2_macos-arm64_2.6.2-0242_COREML_arm64_cli/starnet2" --version
codesign --verify --deep --strict "$HOME/Applications/StarNet2/starnet2_macos-arm64_2.6.2-0242_COREML_arm64_cli/starnet2"
```

Siril stores the executable path in `[core] starnet_exe` in
`~/Library/Application Support/org.siril.Siril/siril/config.1.4.ini` on this Mac.
The missing executable setting caused the earlier M42 failure. No Python,
TensorFlow, or PyTorch dependency was added to the project's environment.

## Retry and check outputs

After fixing setup, reuse the retained exposure stacks:

```bash
uv run run-job jobs/M42.json --stage compose --log
```

For LRGB jobs with star removal enabled, inspect both `lrgb_autostretch_starless`
and `lrgb_veralux_starless` exports (FITS, TIFF, and JPEG). The baseline
`lrgb_autostretch` and `lrgb_veralux` images retain their original stars;
`*_starcomposer` outputs add the separated stars back. A successful version
check alone does not establish that image processing works.

If StarNet fails while `starnet_enabled` is true, the job now raises an error
instead of skipping requested outputs or labeling an original image starless.
Previously saved baseline images remain available, but the job is not reported
as successful or archived as a completed run. If star removal is intentionally
unwanted, set `starnet_enabled` to `false` in the job's options.

Keep generated files until the user has reviewed and approved the output. Then
use the existing `manage-outputs clean` command; never delete original inputs.
