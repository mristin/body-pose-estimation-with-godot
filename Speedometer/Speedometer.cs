using Godot;
using System;

public partial class Speedometer : Node2D
{
    private Line2D _needle = null!;
    private Vector2 _needleFixedPoint;
    private float _needleLength;
    private const float _minAngle = -Mathf.Pi * 3.0f / 4.0f; // -135 degrees
    private const float _maxAngle = Mathf.Pi * 3.0f / 4.0f;  // 135 degrees


    public override void _Ready()
    {
        _needle = GetNode<Line2D>("Needle");
        _needleFixedPoint = _needle.Points[0];

        Vector2 initialEndPoint = _needle.Points[1];
        _needleLength = _needleFixedPoint.DistanceTo(initialEndPoint);

        SetNeedle(1.0f);
    }

    public void SetNeedle(float value)
    {
        // Clamp value to [0, 1] range
        value = Mathf.Clamp(value, 0.0f, 1.0f);

        float angle = Mathf.Lerp(_minAngle, _maxAngle, value);

        Vector2 needleEndPoint = _needleFixedPoint + new Vector2(
            _needleLength * Mathf.Sin(angle),
            _needleLength * -Mathf.Cos(angle)
        );

        _needle.SetPointPosition(0, _needleFixedPoint);
        _needle.SetPointPosition(1, needleEndPoint);
    }
}
