using Godot;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ErgoSkiing
{

    public static class Player
    {
        public const string A = "A";
        public const string B = "B";
    }

    public static class Hand
    {
        public const string Left = "Left";
        public const string Right = "Right";
    }


    namespace Impl
    {

        /// <summary>Enumerate COCO 17 keypoints.</summary>
        public enum KeypointIndex
        {
            Nose = 0,
            LeftEye = 1,
            RightEye = 2,
            LeftEar = 3,
            RightEar = 4,
            LeftShoulder = 5,
            RightShoulder = 6,
            LeftElbow = 7,
            RightElbow = 8,
            LeftWrist = 9,
            RightWrist = 10,
            LeftHip = 11,
            RightHip = 12,
            LeftKnee = 13,
            RightKnee = 14,
            LeftAnkle = 15,
            RightAnkle = 16
        }

        /// <summary>Capture a single body keypoint.</summary>
        public class Keypoint
        {
            public float X;
            public float Y;
            public float Score;

            public Keypoint(float x, float y, float score)
            {
                X = x;
                Y = y;
                Score = score;
            }
        }

        /// <summary>Represent a body pose of a person.</summary>
        public class Pose
        {
            public Keypoint[] Keypoints;

            public Pose(Keypoint[] keypoints)
            {
                Keypoints = keypoints;
            }

            public (Vector2?, Vector2?) MaybeLeftRightHandPositions(
                float keypointScoreThreshold
            )
            {
                Vector2? leftHandPosition = null;
                Vector2? rightHandPosition = null;

                // We use the inverted keypoint index since the image is flipped.
                var leftHand = Keypoints[(int)KeypointIndex.RightWrist];

                if (leftHand.Score > keypointScoreThreshold)
                {
                    leftHandPosition = new Vector2(leftHand.X, leftHand.Y);
                }

                // We use the inverted keypoint index since the image is flipped.
                var rightHand = Keypoints[(int)KeypointIndex.LeftWrist];

                if (rightHand.Score > keypointScoreThreshold)
                {
                    rightHandPosition = new Vector2(rightHand.X, rightHand.Y);
                }

                return (leftHandPosition, rightHandPosition);
            }
        }

        /// <summary> Represent a multi-person pose estimation.</summary>
        public class Inference
        {
            public PlayerPoses PlayerPoses;
            public readonly Image TheImage;

            public Inference(PlayerPoses playerPoses, Image image)
            {
                PlayerPoses = playerPoses;
                TheImage = image;
            }

            public override String ToString()
            {
                var sb = new System.Text.StringBuilder();

                if (PlayerPoses.Left != null)
                {
                    sb.AppendLine("Left Player:");
                    var keypoints = PlayerPoses.Left.Keypoints;
                    for (int i = 0; i < keypoints.Length; i++)
                    {
                        var kp = keypoints[i];
                        sb.AppendLine($"  {i}: {kp.X:F2}, {kp.Y:F2}, {kp.Score:F2}");
                    }
                }

                if (PlayerPoses.Right != null)
                {
                    sb.AppendLine("Right Player:");
                    var keypoints = PlayerPoses.Right.Keypoints;
                    for (int i = 0; i < keypoints.Length; i++)
                    {
                        var kp = keypoints[i];
                        sb.AppendLine($"  {i}: {kp.X:F2}, {kp.Y:F2}, {kp.Score:F2}");
                    }
                }

                return sb.ToString();
            }
        }

        public class PlayerPoses
        {
            public Pose? Left { get; }
            public Pose? Right { get; }

            public PlayerPoses(Pose? left, Pose? right)
            {
                Left = left;
                Right = right;
            }
        }

        public static class PoseToPlayerSuppression
        {
            /// <summary>
            /// Compute for each half of the screen which is the dominant pose.
            /// </summary>
            public static PlayerPoses Apply(
                ICollection<Pose> poses,
                float keypointScoreThreshold
            )
            {
                if (poses.Count == 0)
                {
                    return new PlayerPoses(null, null);
                }

                var leftHalfPoses = new List<ScoredPose>();
                var rightHalfPoses = new List<ScoredPose>();

                foreach (var pose in poses)
                {
                    var poseCenter = ComputePoseCenter(pose, keypointScoreThreshold);
                    if (poseCenter.HasValue)
                    {
                        var poseScore = ComputePoseScore(pose, keypointScoreThreshold);
                        var scoredPose = new ScoredPose(pose, poseScore);

                        if (poseCenter.Value.X < 0.5f)
                        {
                            leftHalfPoses.Add(scoredPose);
                        }
                        else
                        {
                            rightHalfPoses.Add(scoredPose);
                        }
                    }
                }

                Pose? leftPose = null;
                Pose? rightPose = null;

                if (leftHalfPoses.Count > 0)
                {
                    var bestLeft = leftHalfPoses.OrderByDescending(p => p.Score).First();
                    leftPose = bestLeft.Pose;
                }

                if (rightHalfPoses.Count > 0)
                {
                    var bestRight = (
                        rightHalfPoses
                            .OrderByDescending(p => p.Score)
                            .First()
                    );
                    rightPose = bestRight.Pose;
                }

                return new PlayerPoses(leftPose, rightPose);
            }

            /// <summary>
            /// Compute the center position of a pose based on confident keypoints.
            /// </summary>
            private static (float X, float Y)? ComputePoseCenter(
                Pose pose,
                float keypointScoreThreshold
            )
            {
                if (pose.Keypoints.Length == 0)
                    return null;

                float sumX = 0f;
                float sumY = 0f;
                int count = 0;

                foreach (var kp in pose.Keypoints)
                {
                    if (kp.Score >= keypointScoreThreshold)
                    {
                        sumX += kp.X;
                        sumY += kp.Y;
                        count++;
                    }
                }

                return count > 0 ? (sumX / count, sumY / count) : null;
            }

            /// <summary>
            /// Compute a pose confidence score from its confident keypoints.
            /// </summary>
            private static float ComputePoseScore(
                Pose pose,
                float keypointScoreThreshold
            )
            {
                if (pose.Keypoints.Length == 0)
                {
                    return 0f;
                }

                float sum = 0f;
                float count = 0f;

                foreach (var kp in pose.Keypoints)
                {
                    if (kp.Score >= keypointScoreThreshold)
                    {
                        sum += kp.Score;
                        count++;
                    }
                }

                return count > 0f ? sum / count : 0f;
            }

            private sealed class ScoredPose
            {
                public Pose Pose { get; }
                public float Score { get; }

                public ScoredPose(Pose pose, float score)
                {
                    Pose = pose;
                    Score = score;
                }
            }
        }

        /// <summary>Represent a position with timestamp for speed calculation.</summary>
        public class TimestampedPosition
        {
            public Vector2 Position { get; }
            public float Time { get; }

            public TimestampedPosition(Vector2 position, float time)
            {
                Position = position;
                Time = time;
            }
        }

        public class InertialVelocity
        {
            private float _v;
            private readonly float _tau;

            public float Get()
            {
                return _v;
            }

            public InertialVelocity()
            {
                _tau = 0.5f;  // The higher, the less inertia.
                _v = 0f;
            }

            public void Update(float velocity, float delta)
            {
                if (delta <= 0f)
                {
                    return;
                }

                float alpha = (_tau > 0f)
                    ? 1f - (float)Math.Exp(-delta / _tau)
                    : 1f;

                _v += alpha * (velocity - _v);
            }
        }

        public class Smoother
        {
            private Vector2? _lastObservedPosition;
            private Vector2? _smoothedPosition;
            private float _alpha = 0.0f;
            private float _timeSinceLastObservation = 0.0f;
            private CircularBuffer<TimestampedPosition> _positionHistory = (
                new CircularBuffer<TimestampedPosition>(5)
            );

            private float _currentTime = 0.0f;

            // NOTE (mristin):
            // We decay alpha by this amount per second.
            private const float _alphaDecayRate = 0.7f;

            // NOTE (mristin):
            // This determines the smoothing factor of the movement
            // (0 = instant, 1 = no movement).
            private const float _smoothingFactor = 0.3f;

            private readonly InertialVelocity _inertialVelocity = new InertialVelocity();

            public void Observe(Vector2 position, float delta)
            {
                _currentTime += delta;
                _lastObservedPosition = position;
                _alpha = 1.0f;
                _timeSinceLastObservation = 0.0f;

                if (_smoothedPosition.HasValue)
                {
                    // NOTE (mristin):
                    // We smooth the movement using linear interpolation.
                    _smoothedPosition = _smoothedPosition.Value.Lerp(
                        position,
                        1.0f - _smoothingFactor
                    );
                }
                else
                {
                    // NOTE (mristin):
                    // If this is the first observation, we set it directly.
                    _smoothedPosition = position;
                }

                _positionHistory.Add(
                    new TimestampedPosition(_smoothedPosition.Value, _currentTime)
                );

                _inertialVelocity.Update(EstimateVerticalSpeed() ?? 0.0f, delta);
            }

            public (Vector2? position, float alpha, float verticalSpeed) Get()
            {
                return (_smoothedPosition, _alpha, _inertialVelocity.Get());
            }

            public void UpdateWithoutObservation(float delta)
            {
                _currentTime += delta;

                if (_lastObservedPosition.HasValue)
                {
                    _timeSinceLastObservation += delta;

                    if (_timeSinceLastObservation >= 0.2f)
                    {
                        _alpha = Math.Max(0.0f, _alpha - _alphaDecayRate * delta);

                        // NOTE (mristin):
                        // If alpha reaches 0, we can clear the position.
                        if (_alpha <= 0.0f)
                        {
                            _smoothedPosition = null;
                            _lastObservedPosition = null;
                            _positionHistory.Clear();
                        }
                    }
                }

                _inertialVelocity.Update(EstimateVerticalSpeed() ?? 0.0f, delta);
            }

            /// <summary>
            /// Estimate vertical speed for rowing simulation based on position history.
            /// </summary>
            /// <returns>
            /// Vertical speed in pixels per second, or null if insufficient data.
            /// </returns>
            public float? EstimateVerticalSpeed()
            {
                if (_positionHistory.Count < 2)
                {
                    return null;
                }

                float totalVerticalDisplacement = 0f;
                float totalTime = 0f;

                for (int i = 1; i < _positionHistory.Count; i++)
                {
                    var previous = _positionHistory[i - 1];
                    var current = _positionHistory[i];

                    float timeDelta = current.Time - previous.Time;
                    if (timeDelta > 0f)
                    {
                        totalVerticalDisplacement += Math.Abs(
                            current.Position.Y - previous.Position.Y
                        );
                        totalTime += timeDelta;
                    }
                }

                return totalTime > 0f ? totalVerticalDisplacement / totalTime : 0f;
            }
        }

    }  // namespace Impl

}  // namespace ErgoSkiing

public partial class ErgoSki : Node2D
{
    /// <summary>Signal that the player moved the hand.</summary>
    /// <remarks>
    /// <para>The position lives roughly in [0, 1].</para>
    ///
    /// <para>
    /// If the position is not available, it is given as NaN.
    /// </para>
    /// </remarks>
    [Signal]
    public delegate void PlayerPositionUpdatedEventHandler(
        string player,
        string hand,
        float normalizedPosition
    );

    /// <summary>Signal that the player's speed changed.</summary>
    /// <remarks>
    /// <para>The speed lives roughly in [0, 1].</para>
    ///
    /// <para>
    /// If the hand could not be detected, speed is set to 0.0f.
    /// </para>
    /// </remarks>
    [Signal]
    public delegate void PlayerSpeedUpdatedEventHandler(
        string player,
        string hand,
        float normalizedSpeed
    );

    private Label? _status;
    private CameraTexture? _cameraTexture;
    private ImageTexture? _processedTexture;
    private TextureRect? _cameraPort;

    private readonly object _latestFrameLock = new object();
    private Image? _latestFrame;

    private bool _inferenceRunning = false;
    private System.Threading.Thread? _inferenceThread;
    private InferenceSession? _inferenceSession;
    private readonly object _latestInferenceLock = new object();
    private ErgoSkiing.Impl.Inference? _latestInference;

    // NOTE (mristin):
    // We hard-code these to match the body pose estimation model.
    // See: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo
    private const int _inputWidth = 416;
    private const int _inputHeight = 416;

    // NOTE (mristin):
    // We pre-allocate buffers and images to avoid garbage collection pressure.
    private float[]? _inputDataBuffer;
    private DenseTensor<float>? _inputTensorBuffer;

    private Image? _resizedDisplayImage;
    private Image? _resizedInputImage;

    private const float _keypointScoreThreshold = 0.8f;

    private PackedScene? _fireballScene;
    private Fireball _playerAFireballLeft = null!;
    private Fireball _playerAFireballRight = null!;
    private Fireball _playerBFireballLeft = null!;
    private Fireball _playerBFireballRight = null!;

    private ErgoSkiing.Impl.Smoother _playerALeftSmoother = (
        new ErgoSkiing.Impl.Smoother()
    );
    private ErgoSkiing.Impl.Smoother _playerARightSmoother = (
        new ErgoSkiing.Impl.Smoother()
    );
    private ErgoSkiing.Impl.Smoother _playerBLeftSmoother = (
        new ErgoSkiing.Impl.Smoother()
    );
    private ErgoSkiing.Impl.Smoother _playerBRightSmoother = (
        new ErgoSkiing.Impl.Smoother()
    );

    private readonly Dictionary<
        (string player, string hand),
        float?
    > _lastEmittedPositions = (
        new Dictionary<(string, string), float?>()
    );

    private readonly Dictionary<
        (string player, string hand),
        float
    > _lastEmittedSpeeds = (
        new Dictionary<(string, string), float>()
    );

    public override void _Ready()
    {
        GetNode<Line2D>("Panel/Separator").Visible = false;

        _status = GetNode<Label>("Panel/Status");
        _status.Text = "Setting up...";
        CameraServer.MonitoringFeeds = true;

        GetNode<Timer>("SetUpTimer").Timeout += OnSetUpTimerTimeout;

        _fireballScene = GD.Load<PackedScene>($"res://ErgoSki/Fireball/Fireball.tscn");

        var coreColor = new Color(1.0f, 0.6f, 2.0f, 1.0f);
        var playerAColor = new Color(1.0f, 0.0f, 0.0f, 1.0f);
        var playerBColor = new Color(0.0f, 0.0f, 1.0f, 1.0f);

        _playerAFireballLeft = _fireballScene.Instantiate<Fireball>();
        _playerAFireballLeft.Visible = false;
        _playerAFireballLeft.SetColors(
            coreColor,
            playerAColor
        );
        AddChild(_playerAFireballLeft);

        _playerAFireballRight = _fireballScene.Instantiate<Fireball>();
        _playerAFireballRight.Visible = false;
        _playerAFireballRight.SetColors(
            coreColor,
            playerAColor
        );
        AddChild(_playerAFireballRight);

        _playerBFireballLeft = _fireballScene.Instantiate<Fireball>();
        _playerBFireballLeft.Visible = false;
        _playerBFireballLeft.SetColors(
            coreColor,
            playerBColor
        );
        AddChild(_playerBFireballLeft);

        _playerBFireballRight = _fireballScene.Instantiate<Fireball>();
        _playerBFireballRight.Visible = false;
        _playerBFireballRight.SetColors(
            coreColor,
            playerBColor
        );
        AddChild(_playerBFireballRight);

        GD.Print("_Ready done.");
    }

    private void OnSetUpTimerTimeout()
    {
        GD.Print("SetUpTimer timed out.");
        if (CameraServer.GetFeedCount() == 0)
        {
            _status!.Text = "No camera feeds found.";
            return;
        }

        if (_status != null)
        {
            _status.Text = $"Found {CameraServer.GetFeedCount()} camera feed(s).";
        }

        var feed = CameraServer.GetFeed(0);
        var formats = feed.GetFormats();

        var formatSet = false;
        for (int i = 0; i < formats.Count; i++)
        {
            var format = (Godot.Collections.Dictionary)formats[i];

            int width = (int)format["width"];
            int height = (int)format["height"];
            string pixelFormat = format["format"].ToString();

            if (
                width == 640
                && height == 480
                && pixelFormat.StartsWith("YUYV")
            )
            {
                GD.Print($"Set format: {format}");
                feed.SetFormat(i, new Godot.Collections.Dictionary());
                formatSet = true;
                break;
            }
        }

        if (!formatSet)
        {
            _status!.Text = $"No 640x480 YUYV input found.";
            return;
        }

        feed.FeedIsActive = true;

        _cameraTexture = new CameraTexture
        {
            CameraFeedId = feed.GetId(),
            CameraIsActive = true
        };

        _processedTexture = new ImageTexture();

        _cameraPort = GetNode<TextureRect>("Panel/CameraPort");
        _cameraPort!.Texture = _processedTexture;

        feed.FrameChanged += OnFrameChanged;
        _status!.Visible = false;

        GD.Print("Creating the inference session ...");
        {
            string path = $"res://ErgoSki/model/end2end.onnx";

            if (!FileAccess.FileExists(path))
            {
                GD.PushError($"File not found: {path}");
                return;
            }

            // Open file
            using var file = FileAccess.Open(path, FileAccess.ModeFlags.Read);
            if (file == null)
            {
                GD.PushError($"Failed to open file: {path}");
                return;
            }

            // Read all bytes
            byte[] data = file.GetBuffer((long)file.GetLength());

            GD.Print($"Loaded ONNX file, size: {data.Length / 1024.0 / 1024.0:F2} Mb");

            _inferenceSession = new InferenceSession(data);

            // NOTE (mristin):
            // We pre-allocate the buffers and images to avoid the GC pressure.

            _inputDataBuffer = new float[1 * 3 * _inputHeight * _inputWidth];
            _inputTensorBuffer = new DenseTensor<float>(
                _inputDataBuffer,
                new[] { 1, 3, _inputHeight, _inputWidth }
            );

            // NOTE (mristin):
            // We will resize this image later as necessary.
            _resizedDisplayImage = Image.CreateEmpty(
                1, 1, false, Image.Format.Rgb8
            );

            _resizedInputImage = Image.CreateEmpty(
                _inputWidth, _inputHeight, false, Image.Format.Rgb8
            );

            GD.Print("Inference session created.");
        }

        GD.Print("Starting the inference thread...");
        {
            _inferenceRunning = true;
            _inferenceThread = new System.Threading.Thread(InferenceLoop);
            _inferenceThread.Start();
            GD.Print("Inference thread started.");
        }

        GetNode<Line2D>("Panel/Separator").Visible = true;
    }

    public override void _ExitTree()
    {
        _inferenceRunning = false;
        if (_inferenceThread != null)
        {
            _inferenceThread.Join();
            _inferenceThread = null;
        }

        _inferenceSession?.Dispose();
        _inferenceSession = null;
    }

    /// <summary>
    /// Resize and flip an image in a single pass, writing directly to
    /// the target buffer.
    /// </summary>
    /// <param name="sourceImage">Source image to resize and flip</param>
    /// <param name="targetImage">Pre-allocated target image buffer</param>
    private static void ResizeAndFlipInPlace(Image sourceImage, Image targetImage)
    {
        if (sourceImage.GetFormat() != Image.Format.Rgb8)
        {
            throw new InvalidOperationException("Source image must be in RGB8 format");
        }

        Vector2I sourceSize = sourceImage.GetSize();
        if (sourceSize.X == 0 || sourceSize.Y == 0)
        {
            throw new InvalidOperationException(
                $"Source image must not be empty, but got size {sourceSize}"
            );
        }

        if (targetImage.GetFormat() != Image.Format.Rgb8)
        {
            throw new InvalidOperationException("Target image must be in RGB8 format");
        }

        Vector2I targetSize = targetImage.GetSize();
        if (targetSize.X <= 0 || targetSize.Y <= 0)
        {
            throw new InvalidOperationException(
                "Target image dimensions must be greater than zero"
            );
        }

        byte[] sourceData = sourceImage.GetData();

        byte[] targetData = targetImage.GetData();

        int targetWidth = targetSize.X;
        int targetHeight = targetSize.Y;

        float scaleX = (float)sourceSize.X / targetWidth;
        float scaleY = (float)sourceSize.Y / targetHeight;

        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                // Map to source coordinates
                float srcX = x * scaleX;
                float srcY = y * scaleY;

                // Bilinear interpolation coordinates
                int x0 = (int)srcX;
                int y0 = (int)srcY;
                int x1 = Math.Min(x0 + 1, sourceSize.X - 1);
                int y1 = Math.Min(y0 + 1, sourceSize.Y - 1);

                float fx = srcX - x0;
                float fy = srcY - y0;

                // Sample four neighboring pixels
                int idx00 = (y0 * sourceSize.X + x0) * 3;
                int idx01 = (y0 * sourceSize.X + x1) * 3;
                int idx10 = (y1 * sourceSize.X + x0) * 3;
                int idx11 = (y1 * sourceSize.X + x1) * 3;

                // Flip X coordinate for output
                int flippedX = targetWidth - 1 - x;
                int targetIdx = (y * targetWidth + flippedX) * 3;

                // Bilinear interpolation for each color channel
                for (int c = 0; c < 3; c++)
                {
                    float p00 = sourceData[idx00 + c];
                    float p01 = sourceData[idx01 + c];
                    float p10 = sourceData[idx10 + c];
                    float p11 = sourceData[idx11 + c];

                    float p0 = p00 * (1 - fx) + p01 * fx;
                    float p1 = p10 * (1 - fx) + p11 * fx;
                    float p = p0 * (1 - fy) + p1 * fy;

                    // NOTE (mristin):
                    // (int)(p + 0.5f) is an optimization to Mathf.Round.
                    targetData[targetIdx + c] = (byte)(int)(p + 0.5f);
                }
            }
        }

        targetImage.SetData(
            targetSize.X,
            targetSize.Y,
            false,
            Image.Format.Rgb8,
            targetData
        );
    }

    private void OnFrameChanged()
    {
        Image image = _cameraTexture!.GetImage();
        if (image == null)
        {
            GD.Print("No image received from camera texture.");
            return;
        }

        lock (_latestFrameLock)
        {
            _latestFrame = image;
        }

        // NOTE (mristin):
        // We have to be robust to 0x0 camera port so we only update the image
        // if it is not empty.
        Vector2I targetSize = (Vector2I)_cameraPort!.Size;
        if (targetSize.X > 0 && targetSize.Y > 0)
        {
            // NOTE (mristin):
            // We re-use the pre-allocated image for performance.
            if (_resizedDisplayImage!.GetSize() != targetSize)
            {
                _resizedDisplayImage = Image.CreateEmpty(
                    targetSize.X, targetSize.Y, false, Image.Format.Rgb8
                );
            }

            ResizeAndFlipInPlace(image, _resizedDisplayImage);

            if (_processedTexture!.GetWidth() == 0)
            {
                _processedTexture.SetImage(_resizedDisplayImage);
            }
            else
            {
                _processedTexture.Update(_resizedDisplayImage);
            }
        }
    }

    private void InferenceLoop()
    {
        GD.Print("Inference loop started.");
        while (_inferenceRunning)
        {
            Image? frame = null;
            lock (_latestFrameLock)
            {
                if (_latestFrame != null)
                {
                    frame = _latestFrame;
                    _latestFrame = null;
                }
            }

            if (frame == null || _inferenceSession == null)
            {
                System.Threading.Thread.Sleep(1);
                continue;
            }

            if (_resizedInputImage == null)
            {
                throw new InvalidOperationException(
                    "Expected the resized input image to have been pre-allocated, " +
                    "but _resizedInputImage is null."
                );
            }

            ResizeAndFlipInPlace(frame, _resizedInputImage);

            // NOTE (mristin):
            // We use bulk pixel data extraction instead of GetPixel() for performance.
            // RTMO expects simple 0-255 range without normalization.
            byte[] pixelData = _resizedInputImage.GetData();

            int pixelCount = _inputWidth * _inputHeight;
            int rOffset = 0;
            int gOffset = pixelCount;
            int bOffset = 2 * pixelCount;

            // NOTE (mristin):
            // We convert from RGB8 format (3 bytes per pixel) to planar float format.
            for (int i = 0; i < pixelCount; i++)
            {
                int pixelIndex = i * 3; // RGB8 has 3 bytes per pixel.

                // NOTE (mristin):
                // While GetPixel gives [0,1], GetData() already gives us [0, 255] so
                // no normalization is necessary.

                _inputDataBuffer![rOffset + i] = pixelData[pixelIndex];     // R
                _inputDataBuffer![gOffset + i] = pixelData[pixelIndex + 1]; // G
                _inputDataBuffer![bOffset + i] = pixelData[pixelIndex + 2]; // B
            }

            // NOTE (mristin):
            // We reuse the pre-allocated tensor buffer instead of creating a new one.

            //// NOTE (mristin):
            //// We pick the input name dynamically to be more robust to model
            //// change in the future.
            string inputName = _inferenceSession.InputMetadata.Keys.First();

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, _inputTensorBuffer!)
            };

            using var results = _inferenceSession.Run(inputs);

            String expectedOutputName = "keypoints";
            int keypointsIndex = -1;
            for (var i = 0; i < results.Count(); i++)
            {
                if (results[i].Name == expectedOutputName)
                {
                    keypointsIndex = i;
                    break;
                }
            }

            if (keypointsIndex == -1)
            {
                throw new InvalidOperationException(
                    $"No output {expectedOutputName} in the outputs."
                );
            }

            var output = results[keypointsIndex].AsTensor<float>();
            var dims = output.Dimensions.ToArray();

            // NOTE (mristin):
            // We expect: [1, N, K, 3].
            int numPersons = dims[1];
            int numKeypoints = dims[2];

            ICollection<ErgoSkiing.Impl.Pose> poses = new List<ErgoSkiing.Impl.Pose>();

            for (int p = 0; p < numPersons; p++)
            {
                ErgoSkiing.Impl.Keypoint[] keypoints = (
                    new ErgoSkiing.Impl.Keypoint[numKeypoints]
                );

                int validKeypoints = 0;

                for (int k = 0; k < numKeypoints; k++)
                {
                    float x = output[0, p, k, 0] / (float)_inputWidth;
                    float y = output[0, p, k, 1] / (float)_inputHeight;
                    float score = output[0, p, k, 2];

                    if (score > _keypointScoreThreshold)
                    {
                        validKeypoints += 1;
                    }

                    keypoints[k] = new ErgoSkiing.Impl.Keypoint(x, y, score);
                }

                // NOTE (mristin):
                // We accept only poses with sufficient keypoints.
                if (validKeypoints > 4)
                {
                    poses.Add(new ErgoSkiing.Impl.Pose(keypoints));
                }
            }

            var playerPoses = ErgoSkiing.Impl.PoseToPlayerSuppression.Apply(
                poses,
                _keypointScoreThreshold
            );

            var inference = new ErgoSkiing.Impl.Inference(playerPoses, frame);
            lock (_latestInferenceLock)
            {
                _latestInference = inference;
            }

            // NOTE (mristin):
            // We sleep a bit to avoid clogging the game.
            System.Threading.Thread.Sleep(10);
        }

        GD.Print("Left inference loop.");
    }

    public override void _Process(double delta)
    {
        ErgoSkiing.Impl.Inference? inference = null;

        lock (_latestInferenceLock)
        {
            if (_latestInference != null)
            {
                inference = _latestInference;
                _latestInference = null;
            }
        }

        Vector2? playerALeftPosition = null;
        Vector2? playerARightPosition = null;

        Vector2? playerBLeftPosition = null;
        Vector2? playerBRightPosition = null;

        if (inference != null)
        {
            if (inference.PlayerPoses.Left != null)
            {
                (playerALeftPosition, playerARightPosition) = (
                    inference
                        .PlayerPoses
                        .Left
                        .MaybeLeftRightHandPositions(_keypointScoreThreshold)
                );
            }

            if (inference.PlayerPoses.Right != null)
            {
                (playerBLeftPosition, playerBRightPosition) = (
                    inference
                        .PlayerPoses
                        .Right
                        .MaybeLeftRightHandPositions(_keypointScoreThreshold)
                );
            }
        }

        foreach (
            var (
                keypointPosition,
                smoother,
                fireball,
                player,
                hand
            ) in new (Vector2?, ErgoSkiing.Impl.Smoother, Fireball, string, string)[]
            {
                (
                    playerALeftPosition,
                    _playerALeftSmoother,
                    _playerAFireballLeft,
                    ErgoSkiing.Player.A,
                    ErgoSkiing.Hand.Left
                ),
                (
                    playerARightPosition,
                    _playerARightSmoother,
                    _playerAFireballRight,
                    ErgoSkiing.Player.A,
                    ErgoSkiing.Hand.Right
                ),
                (
                    playerBLeftPosition,
                    _playerBLeftSmoother,
                    _playerBFireballLeft,
                    ErgoSkiing.Player.B,
                    ErgoSkiing.Hand.Left
                ),
                (
                    playerBRightPosition,
                    _playerBRightSmoother,
                    _playerBFireballRight,
                    ErgoSkiing.Player.B,
                    ErgoSkiing.Hand.Right
                )
            }
        )
        {

            if (keypointPosition.HasValue)
            {
                smoother.Observe(keypointPosition.Value, (float)delta);
            }
            else
            {
                smoother.UpdateWithoutObservation((float)delta);
            }

            var (smoothedPosition, alpha, speed) = smoother.Get();

            if (smoothedPosition.HasValue)
            {
                var cameraPortSize = _cameraPort!.Size;

                var position = new Vector2(
                    _cameraPort!.Position.X
                        + smoothedPosition.Value.X * cameraPortSize.X,
                    _cameraPort!.Position.Y
                        + smoothedPosition.Value.Y * cameraPortSize.Y
                );

                if (!fireball.Visible)
                {
                    fireball.Position = position;
                    fireball.Visible = true;
                }
                else
                {
                    fireball.Move(position - fireball.Position);
                }
                fireball.Modulate = new Color(1.0f, 1.0f, 1.0f, alpha);

                fireball.UpdateWithSpeed(speed);
            }
            else
            {
                fireball.Visible = false;
            }

            var key = (player, hand);

            float? normalizedPosition = smoothedPosition?.Y;

            if (
                !_lastEmittedPositions.TryGetValue(key, out var lastPosition)
                || (lastPosition != null) ^ (normalizedPosition != null)
                || (
                    lastPosition != null
                    && normalizedPosition != null
                    && Math.Abs(lastPosition.Value - normalizedPosition.Value) > 0.001f
                )
            )
            {
                _lastEmittedPositions[key] = normalizedPosition;
                EmitSignal(
                    SignalName.PlayerPositionUpdated,
                    player,
                    hand,
                    normalizedPosition ?? float.NaN
                );
            }

            if (
                !_lastEmittedSpeeds.TryGetValue(key, out var lastSpeed)
                || Math.Abs(lastSpeed - speed) > 0.001f
            )
            {
                _lastEmittedSpeeds[key] = speed;
                EmitSignal(
                    SignalName.PlayerSpeedUpdated,
                    player,
                    hand,
                    speed
                );
            }
        }
    }
}
