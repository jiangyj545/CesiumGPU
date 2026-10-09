import { describe, it, expect } from 'vitest';
import { CallbackProperty, resolveProperty, isProperty } from '../../src/dataSource/CallbackProperty';
import { Cartesian3 } from '../../src/math/Cartesian3';

describe('CallbackProperty', () => {
  it('samples getValue through the callback, passing the time argument', () => {
    const seen: unknown[] = [];
    const prop = new CallbackProperty((time?: unknown) => {
      seen.push(time);
      return new Cartesian3(1, 2, 3);
    });
    const v = prop.getValue(42);
    expect(v).toBeInstanceOf(Cartesian3);
    expect(v.x).toBe(1);
    expect(seen).toEqual([42]);
    expect(prop.getValue()).toBeDefined();
    expect(seen[1]).toBeUndefined();
  });

  it('defaults to non-constant and honours the isConstant flag', () => {
    expect(new CallbackProperty(() => 1).isConstant).toBe(false);
    expect(new CallbackProperty(() => 1, true).isConstant).toBe(true);
  });

  it('raises definitionChanged when the callback is replaced', () => {
    const prop = new CallbackProperty(() => 1);
    let raised = 0;
    prop.definitionChanged.addListener(() => raised++);
    expect(raised).toBe(0);
    prop.setCallback(() => 2);
    expect(raised).toBe(1);
    expect(prop.getValue()).toBe(2);
  });
});

describe('resolveProperty', () => {
  it('passes raw values through unchanged', () => {
    const p = new Cartesian3(9, 9, 9);
    expect(resolveProperty(p)).toBe(p);
    expect(isProperty({ getValue: 5 } as unknown as { getValue: number })).toBe(false);
    expect(resolveProperty('plain')).toBe('plain');
  });

  it('samples Properties at the given time', () => {
    const prop = new CallbackProperty((t) => `v${String(t)}`);
    expect(resolveProperty(prop, 7)).toBe('v7');
  });

  it('isProperty only matches objects with a callable getValue', () => {
    expect(isProperty(new CallbackProperty(() => 1))).toBe(true);
    expect(isProperty(new Cartesian3())).toBe(false);
    expect(isProperty(undefined)).toBe(false);
    expect(isProperty({ getValue: 5 } as never)).toBe(false);
  });

  it('accepts any getValue-bearing object (duck-typed Property)', () => {
    const duck = { getValue: (t?: unknown) => new Cartesian3(Number(t), 0, 0) };
    const resolved = resolveProperty<Cartesian3>(duck, 3);
    expect(resolved?.x).toBe(3);
  });
});
