using Godot;
using System;
using System.Collections.Generic;
using System.Linq;

public partial class Fireball : Node2D
{
    [Export] public Color CoreColor { get; set; } = Colors.White;
    [Export] public Color GlowColor { get; set; } = new Color(1.0f, 0.6f, 0.2f, 1.0f);

    private ShaderMaterial _fireballShader = null!;

    private Line2D _trail = null!;

    private ErgoSkiing.CircularBuffer<Vector2> _positionHistory = (
        new ErgoSkiing.CircularBuffer<Vector2>(50)
    );

    public override void _Ready()
    {
        var colorRect = GetNode<ColorRect>("ColorRect");

        var fireballShader = GD.Load<Shader>("res://ErgoSki/Fireball/fireball.gdshader");
        _fireballShader = new ShaderMaterial();
        _fireballShader.Shader = fireballShader;
        _fireballShader.SetShaderParameter("time_scale", 1.0f);
        _fireballShader.SetShaderParameter("core_color", CoreColor);
        _fireballShader.SetShaderParameter("glow_color", GlowColor);

        colorRect.Material = _fireballShader;

        _trail = new Line2D();
        _trail.Width = 8.0f;
        _trail.DefaultColor = GlowColor;
        _trail.EndCapMode = Line2D.LineCapMode.Round;
        _trail.BeginCapMode = Line2D.LineCapMode.Round;
        _trail.JointMode = Line2D.LineJointMode.Round;

        AddChild(_trail);

        // Move trail behind fireball
        MoveChild(_trail, 0);

        UpdateShaderColors();
    }

    private void UpdateShaderColors()
    {
        if (_fireballShader == null || _trail == null)
        {
            return;
        }

        _fireballShader.SetShaderParameter("core_color", CoreColor);
        _fireballShader.SetShaderParameter("glow_color", GlowColor);

        _trail.DefaultColor = GlowColor;
    }

    public void SetColors(Color coreColor, Color glowColor)
    {
        CoreColor = coreColor;
        GlowColor = glowColor;
        UpdateShaderColors();
    }

    /// <summary>Move the fireball relative to its previous position.</summary>
    public void Move(Vector2 diff)
    {
        var oldPosition = Position;
        Position += diff;

        _positionHistory.Add(diff);

        if (_positionHistory.Count < 2)
        {
            _trail.Visible = false;
            return;
        }

        _trail.Visible = true;
        _trail.ClearPoints();

        var cursor = new Vector2(0.0f, 0.0f);
        for (var i = _positionHistory.Count - 1; i >= 0; i--)
        {
            var prevDiff = _positionHistory[i];
            cursor -= prevDiff;

            _trail.AddPoint(cursor);
        }

        _trail.Visible = true;

        //// NOTE (mristin):
        //// We create gradient effect by adjusting alpha along the trail.
        var gradient = new Gradient();

        // NOTE (mristin):
        // We have to start from >0.0f to avoid artefacts causing the end of tail to be
        // black.
        gradient.AddPoint(0.0f, new Color(GlowColor.R, GlowColor.G, GlowColor.B, 0.5f));

        gradient.AddPoint(1.0f, new Color(GlowColor.R, GlowColor.G, GlowColor.B, 0.1f));
        _trail.Gradient = gradient;
    }

    /// <summary>Update fireball with vertical speed for intensity effect.</summary>
    /// <remarks>
    /// The <paramref name="normalizedAbsSpeed" /> is expected to be in [0,1].
    /// </remarks>
    public void UpdateWithSpeed(float? normalizedAbsSpeed)
    {
        if (_fireballShader == null || !normalizedAbsSpeed.HasValue)
        {
            return;
        }

        Scale = Vector2.One * (0.6f + normalizedAbsSpeed.Value * 0.4f);
    }
}
