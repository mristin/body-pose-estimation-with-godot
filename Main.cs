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

    public override void _Ready()
    {
        _ergoSki = GetNode<ErgoSki>("ErgoSki");
        _ergoSki.PlayerSpeedUpdated += OnPlayerSpeedUpdated;

        _speedometerALeft = GetNode<Speedometer>("SpeedometerALeft");
        _speedometerARight = GetNode<Speedometer>("SpeedometerARight");
        _speedometerBLeft = GetNode<Speedometer>("SpeedometerBLeft");
        _speedometerBRight = GetNode<Speedometer>("SpeedometerBRight");
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
}
