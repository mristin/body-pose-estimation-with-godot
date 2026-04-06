using Godot;
using System;
using System.Collections.Generic;
using System.Linq;

public partial class Main : Node2D
{
    private ErgoSki? _ergoSki;
    private Speedometer? _speedometerALeft;
    private Speedometer? _speedometerARight;
    private Speedometer? _speedometerBLeft;
    private Speedometer? _speedometerBRight;

    private VerticalIntensity? _positionALeft;
    private VerticalIntensity? _positionARight;
    private VerticalIntensity? _positionBLeft;
    private VerticalIntensity? _positionBRight;

    private TextEdit? _textA;
    private TextEdit? _textB;

    public override void _Ready()
    {
        _ergoSki = GetNode<ErgoSki>("ErgoSki");

        _positionALeft = GetNode<VerticalIntensity>("PositionALeft");
        _positionARight = GetNode<VerticalIntensity>("PositionARight");
        _positionBLeft = GetNode<VerticalIntensity>("PositionBLeft");
        _positionBRight = GetNode<VerticalIntensity>("PositionBRight");

        _speedometerALeft = GetNode<Speedometer>("SpeedometerALeft");
        _speedometerARight = GetNode<Speedometer>("SpeedometerARight");
        _speedometerBLeft = GetNode<Speedometer>("SpeedometerBLeft");
        _speedometerBRight = GetNode<Speedometer>("SpeedometerBRight");

        _textA = GetNode<TextEdit>("TextA");
        _textB = GetNode<TextEdit>("TextB");

        _ergoSki.PlayerPositionUpdated += OnPlayerPositionUpdated;
        _ergoSki.PlayerSpeedUpdated += OnPlayerSpeedUpdated;
        _ergoSki.PlayerSymbolEmitted += OnPlayerSymbolEmitted;
        _ergoSki.StartTrackingSymbols();
    }

    private void OnPlayerSpeedUpdated(string player, string hand, float normalizedSpeed)
    {
        Speedometer? speedometer = null;

        if (player == ErgoSkiing.Player.A && hand == ErgoSkiing.Hand.Left)
        {
            speedometer = _speedometerALeft;
        }
        else if (player == ErgoSkiing.Player.A && hand == ErgoSkiing.Hand.Right)
        {
            speedometer = _speedometerARight;
        }
        else if (player == ErgoSkiing.Player.B && hand == ErgoSkiing.Hand.Left)
        {
            speedometer = _speedometerBLeft;
        }
        else if (player == ErgoSkiing.Player.B && hand == ErgoSkiing.Hand.Right)
        {
            speedometer = _speedometerBRight;
        }
        else
        {
            throw new ArgumentException(
                $"Unknown player '{player}' or hand '{hand}'"
            );
        }

        if (speedometer == null)
        {
            throw new InvalidOperationException(
                $"Speedometer for player {player} {hand} is not initialized"
            );
        }

        speedometer.SetNeedle(normalizedSpeed);
    }

    private void OnPlayerPositionUpdated(
        string player,
        string hand,
        float normalizedPosition
    )
    {
        VerticalIntensity? verticalIntensity = null;

        if (player == ErgoSkiing.Player.A && hand == ErgoSkiing.Hand.Left)
        {
            verticalIntensity = _positionALeft;
        }
        else if (player == ErgoSkiing.Player.A && hand == ErgoSkiing.Hand.Right)
        {
            verticalIntensity = _positionARight;
        }
        else if (player == ErgoSkiing.Player.B && hand == ErgoSkiing.Hand.Left)
        {
            verticalIntensity = _positionBLeft;
        }
        else if (player == ErgoSkiing.Player.B && hand == ErgoSkiing.Hand.Right)
        {
            verticalIntensity = _positionBRight;
        }
        else
        {
            throw new ArgumentException(
                $"Unknown player '{player}' or hand '{hand}'"
            );
        }

        if (verticalIntensity == null)
        {
            throw new InvalidOperationException(
                $"VerticalIntensity for player {player} {hand} is not initialized"
            );
        }

        verticalIntensity.Set(normalizedPosition);
    }

    public void OnPlayerSymbolEmitted(string player, string symbol)
    {
        TextEdit? textEdit = null;

        if (player == ErgoSkiing.Player.A)
        {
            textEdit = _textA;
        }
        else if (player == ErgoSkiing.Player.B)
        {
            textEdit = _textB;
        }
        else
        {
            throw new ArgumentException($"Unknown player '{player}'");
        }

        if (textEdit == null)
        {
            throw new InvalidOperationException(
                $"TextEdit for player {player} is not initialized"
            );
        }

        textEdit.Text += symbol;
    }
}
