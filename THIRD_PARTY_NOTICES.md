# Third-party components

## llama.cpp conversion sources

`vendor/llama_cpp/` contains conversion sources from
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), revision
`970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc`.

These files are distributed under the upstream MIT license. The copyright and
full license text are preserved in [vendor/llama_cpp/LICENSE](vendor/llama_cpp/LICENSE).
[SOURCE.json](vendor/llama_cpp/SOURCE.json) records the included directories and
revision. Native llama.cpp inference binaries are not bundled.

Python dependencies are installed separately and retain their respective
licenses. Base models and training data are supplied separately by the user.
