using System;
using System.Collections;
using System.Collections.Generic;

public class CircularBuffer<T> : IReadOnlyList<T>
{
    private readonly T[] _buffer;
    private readonly int _capacity;
    private int _count;
    private int _startIndex;

    public CircularBuffer(int capacity)
    {
        _capacity = capacity;
        _buffer = new T[capacity];
        _count = 0;
        _startIndex = 0;
    }

    public int Count => _count;

    public void Add(T item)
    {
        if (_count < _capacity)
        {
            _buffer[_count] = item;
            _count++;
        }
        else
        {
            _buffer[_startIndex] = item;
            _startIndex = (_startIndex + 1) % _capacity;
        }
    }

    public void Clear()
    {
        _count = 0;
        _startIndex = 0;
    }

    public IEnumerator<T> GetEnumerator()
    {
        for (int i = 0; i < _count; i++)
        {
            int index = (_startIndex + i) % _capacity;
            yield return _buffer[index];
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public T this[int index]
    {
        get
        {
            if (index < 0 || index >= _count)
                throw new ArgumentOutOfRangeException(nameof(index));

            int actualIndex = (_startIndex + index) % _capacity;
            return _buffer[actualIndex];
        }
    }
}
