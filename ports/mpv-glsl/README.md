# Experimental mpv GLSL backend for AnimeJaNai

This port executes the original AnimeJaNai HD V3.1 Performance SPANF3 model through mpv's fragment shader API. It is intended for players such as IINA on macOS that can load mpv GLSL shaders but do not include the Windows inference backend. The converted model is checked in as `shaders/AnimeJaNai_Performance.glsl`: no ONNX parsing, Python runtime, conversion, TensorRT or DirectML is used during playback. The graphics driver still compiles GLSL on first use.

The model has not been retrained or pruned. Its original coefficients are stored in immutable RGBA float textures. Activations use four-channel spatial tiles, avoiding a separate sampler per channel group. The passes implement reflected odd-edge padding, pixel unshuffle, convolution with zero padding, ReLU, addition, Tanh, multiplication, channel concatenation, pixel shuffle and the final 2× crop. The port includes native-resolution Fast, Performance, Balanced and Sharp model profiles. It does not include the Manager, frame interpolation or other models.

## Use

Load `shaders/AnimeJaNai_Performance.glsl` as an mpv `glsl-shaders` entry. For a standalone mpv build with the original GPU renderer:

```sh
mpv --vo=gpu --glsl-shaders=/absolute/path/AnimeJaNai_Performance.glsl video.mkv
```

IINA plugins can set the same `glsl-shaders` property through their mpv API. Select a floating-point intermediate buffer format (mpv's normal GPU default on the tested system); normalized integer buffers cannot preserve signed model activations. With the original `vo=gpu` renderer, `fbo-format=rgba16f` is suitable. Other renderers are unverified.

This is an experimental compatibility implementation. It can consume substantial GPU memory, especially on large inputs, and does not promise real-time HD playback or TensorRT-like performance. Start with a small input. The converted representation removes conversion work from playback; it does not remove the neural network's inference work.

## Reproduce conversion

From this directory, use a local Python environment with the build requirements, then regenerate the committed resource. The original ONNX file is read from the existing repository model directory; it is not duplicated here. Provenance and the original model SHA-256 are recorded in `models/animejanai/model.json`. The checksum guard rejects different model files.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-build.txt
.venv/bin/python scripts/compile-animejanai.py
git diff --exit-code -- shaders/AnimeJaNai_Performance.glsl
```

The converter also accepts explicit `--model`, `--metadata` and `--output` paths. A different model requires auditing its graph and adapting the converter; replacing the checksum alone is insufficient.

## Compare against ONNX Runtime on macOS

Install the test requirements and mpv development headers. Link the helper to the libmpv in the target IINA app to test that exact renderer, or substitute the libmpv supplied by the standalone player being tested.

```sh
.venv/bin/python -m pip install -r requirements-test.txt
clang -Wno-deprecated-declarations -I "$(brew --prefix mpv)/include" tests/render-mpv.c -framework OpenGL /Applications/IINA.app/Contents/Frameworks/libmpv.2.dylib -Wl,-rpath,/Applications/IINA.app/Contents/Frameworks -o .venv/render-mpv
.venv/bin/python tests/verify-animejanai.py
```

The helper creates an offscreen OpenGL context, executes the shader with libmpv's OpenGL rendering API, reads float output pixels and compares them to CPU ONNX Runtime. It rejects renderer compilation errors as well as numerical mismatches. It checks even dimensions (32×24), odd dimensions (33×25), and a larger 256×144 input. `AJN_FBO_FORMAT=rgba32f` selects full-precision intermediate buffers for diagnostics.

On the tested Apple M1 with IINA's libmpv and normal half-float intermediates, mean absolute RGB error was approximately 0.001 and maximum absolute error was below 0.008, on normalized input/output values. The output was exactly twice the source dimensions. A separate 320×180 FFV1 motion smoke test rendered 12 successive frames at 640×360 without GL errors. Set `AJN_TEST_PLAYBACK=1` when running the rendering helper with a video to exercise consecutive frames. Detailed observations are in `verification.json`. These cases validate shader compilation, model operations and odd-edge behavior; they are not HD real-time benchmarks or a GUI playback test.

## Attribution and license

The original model is by the-database, from upstream revision `545ac129cf189ac5a68101d9f3d91c4d0fc78217`. The model and the converted coefficients retain CC BY-NC-SA 4.0; the original license is preserved in `models/animejanai/UPSTREAM-LICENSE`. Conversion is a change of execution representation only. The converter and verification helpers are by yorkyang2333 and are MIT-licensed under this directory's `LICENSE`. No existing upstream playback code or default configuration is modified.

## Native model profiles

AnimeEnhancer selects among native-resolution preconverted models; it does not resize the input to create a speed tier. `AnimeJaNai_Fast.glsl` uses the upstream V3 `SuperUltraCompact` model. `AnimeJaNai_Performance.glsl` uses V3.1 Performance, `AnimeJaNai_Balanced.glsl` uses V3.1 Balanced, and `AnimeJaNai_Sharp.glsl` uses V3.1 Sharp1 Performance. All four receive the same source resolution and produce 2× output. The heavier models are quality references rather than real-time profiles on an M1.

On the tested M1 with a 1920×1080 source and the correct 3840×2160 2× render target, Fast took approximately 278 ms/frame (about 3.6 FPS) after shader compilation. A separate 1920×1080 render target measured about 29 ms/frame, but that target does not exercise the model’s full 2× output and is not a normal playback benchmark. The heavier profiles are slower still. Use Fast only when the render target is limited; the other profiles are useful for screenshots, short clips or quality comparison.
