import assert from 'node:assert/strict';
import { test } from 'node:test';
import { webcrypto } from 'node:crypto';
globalThis.crypto ??= webcrypto;
import { SCENARIOS, runSimulation, traceToJsonl } from './simulator.mjs';

test('has four bounded local scenarios', () => assert.deepEqual(SCENARIOS.map((item) => item.id), ['nominal', 'fan-degradation', 'blocked-path', 'frozen-sensor']));
test('replays identically and never grants authority', async () => {
  const first = await runSimulation({ scenarioId: 'fan-degradation', seed: 42, ticks: 24 });
  const second = await runSimulation({ scenarioId: 'fan-degradation', seed: 42, ticks: 24 });
  assert.deepEqual(first, second); assert.match(first.digest, /^[a-f0-9]{64}$/);
  assert.ok(first.trace.every((row) => row.actionAuthority === 'none'));
});
test('exports ordered JSONL and freezes the fixture sensor', async () => {
  const result = await runSimulation({ scenarioId: 'frozen-sensor', seed: 7, ticks: 12 });
  assert.equal(result.trace[8].reportedAirflow, result.trace[7].reportedAirflow);
  assert.equal(traceToJsonl(result).trim().split('\n').length, 12);
});
