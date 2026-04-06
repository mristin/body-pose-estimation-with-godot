using Godot;
using System;

[Tool]
public partial class VerticalIntensity : Node2D
{
    private ColorRect? _background;
    private ColorRect? _indicator;
    private Color _color = Colors.Blue;

    [Export]
    public Color Color
    {
        get => _color;
        set
        {
            _color = value;
            UpdateIndicatorColor();
        }
    }

    public override void _Ready()
    {
        _background = GetNode<ColorRect>("Background");
        _indicator = GetNode<ColorRect>("Indicator");
        UpdateIndicatorColor();
    }

    private void UpdateIndicatorColor()
    {
        // Try to get the indicator if we don't have it yet (for editor)
        if (_indicator == null && HasNode("Indicator"))
        {
            _indicator = GetNode<ColorRect>("Indicator");
        }

        if (_indicator != null)
        {
            _indicator.Color = Color;
        }
    }

    public void Set(float value)
    {
        if (float.IsNaN(value))
        {
            _indicator!.Color = new Color(Color.R, Color.G, Color.B, 0.3f);
            return;
        }

        _indicator!.Color = Color;

        value = Mathf.Clamp(value, 0.0f, 1.0f);

        float newHeight = 130.0f * Mathf.Round(value * 10.0f) / 10.0f;

        _indicator!.Size = new Vector2(10, newHeight);
        _indicator!.Position = new Vector2(5, 5 + (130.0f - newHeight));
    }
}
