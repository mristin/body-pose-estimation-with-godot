# body-pose-estimation-with-godot

[![Continuous integration](https://github.com/mristin/body-pose-estimation-with-godot/actions/workflows/ci.yaml/badge.svg)](https://github.com/mristin/body-pose-estimation-with-godot/actions/workflows/ci.yaml)

This is a simple demo code where we tried to get body pose estimation working in Godot.

[![Screenshot](https://github.com/mristin/body-pose-estimation-with-godot/raw/main/screenshot.png)](https://youtu.be/2Ju0RFVwm-s)

The demo already includes inertia for having stable speed in case you want to make ergo skiing games.
There is also a rudimentary symbol recognition based on the hand pose.

We used C# since we resort to ONNX runtime to run model inference.

The body pose recognition code is encapsulated in [ErgoSki/](ErgoSki/) scene which you can readily copy to your project and use.

## Acknowledgments

We used [RTMO-t model].

[RTMO model]: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo

Speedometer image is taken from https://opengameart.org/content/speedometer-0.

## Contributing

Restore the development tools:

```
dotnet restore
```

Re-format before the pull request:

```
dotnet format
```

### Commit Messages

We follow Chris Beams' [guidelines on commit messages]:

1) Separate subject from body with a blank line
2) Limit the subject line to 50 characters
3) Capitalize the subject line
4) Do not end the subject line with a period
5) Use the imperative mood in the subject line
6) Wrap the body at 72 characters
7) Use the body to explain *what* and *why* vs. *how*

[guidelines on commit messages]: https://chris.beams.io/posts/git-commit/
