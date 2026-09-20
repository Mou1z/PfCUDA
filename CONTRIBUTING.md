# Contributing to PfCUDA

Bug reports, questions and pull requests are all welcome.

## Reporting a problem

Open an issue at
[github.com/Mou1z/PfCUDA/issues](https://github.com/Mou1z/PfCUDA/issues). The
single most useful thing you can include is the output of:

```bash
./dev doctor
```

It reports your GPU, CUDA toolkit, compiler, Python and library versions, and
which copy of `pfcuda` is being imported — which covers most of what anyone
would otherwise have to ask you.

For a wrong numerical result, please also include the matrix size, the dtype,
and which of the four entry points you called.

## Getting set up

You need Linux or WSL2, an NVIDIA GPU, and the CUDA Toolkit. Then:

```bash
git clone https://github.com/Mou1z/PfCUDA.git
cd PfCUDA
./dev
```

The first run creates a virtual environment, works out whether your driver
needs the CUDA 12 or CUDA 13 build of JAX, installs everything and compiles the
kernels. That takes a few minutes, mostly downloading. After that a rebuild
takes about five seconds.

| Command | What it does |
| --- | --- |
| `./dev` | Rebuild and run the fast tests (~17 s) |
| `./dev test` | Rebuild and run the whole suite (~2 min) |
| `./dev bench` | Run the benchmarks |
| `./dev doctor` | Report on your environment; changes nothing |
| `./dev clean` | Delete build outputs |

On Windows, run `.\dev.ps1 <command>` from PowerShell; it forwards into WSL2.

## Making a change

Please make sure `./dev test` passes before opening a pull request. If you
change a CUDA kernel, `./dev bench --compare` will show you how the timings
moved against the stored baseline.

Tests are marked `gpu` when they need a CUDA device and `slow` when they take a
while. Continuous integration has no GPU, so it runs `pytest -m "not gpu"`,
builds against both CUDA 12 and CUDA 13, and checks that the package installs
from a source distribution on a machine with no `nvcc` on `PATH`.

Some things worth knowing before you dig in:

- **Everything must run in float64 unless the test says otherwise.** JAX
  quietly falls back to float32 unless x64 mode is enabled, which is faster and
  far less accurate.
- **The GPU path splits at 32x32.** `pfaffian()` handles matrices up to that
  size; `slog_pfaffian()` handles 34x34 and above. They use different kernels.
- **The two CPU backends overwrite the matrix you give them** for `n > 4`.

If you are changing the benchmarks, `benchmarking/README.md` explains the
measurement protocol and why it is set up that way.

## Style

Match the code already there. Comments should explain why something is done,
not restate what the line does.

## Licence

Contributions are accepted under the [MIT Licence](LICENSE), the same terms as
the rest of the project.
