export const SCENARIOS = Object.freeze([
  { id: 'nominal', label: 'Nominal circulation', description: 'Baseline airflow with bounded ambient variation.' },
  { id: 'fan-degradation', label: 'Fan degradation', description: 'A fixed efficiency decline reduces supplied airflow.' },
  { id: 'blocked-path', label: 'Blocked path', description: 'A fixed duct restriction raises the airflow residual.' },
  { id: 'frozen-sensor', label: 'Frozen sensor', description: 'The reported airflow is held after its fixed fault point.' },
]);
const FIXTURES = Object.freeze({ nominal: [1, 0], 'fan-degradation': [.62, 0], 'blocked-path': [1, .48], 'frozen-sensor': [1, 0, 8] });
const round = (value) => Number(value.toFixed(4));
function rng(seed) { let state = (seed >>> 0) || 1; return () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 4294967296; }; }
export function traceToJsonl(result) { return `${result.trace.map((row) => JSON.stringify(row)).join('\n')}\n`; }
export async function sha256(value) {
  const bytes = new TextEncoder().encode(value);
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}
export async function runSimulation({ scenarioId, seed, ticks = 72 }) {
  if (!Object.hasOwn(FIXTURES, scenarioId)) throw new Error('Unknown closed-schema scenario.');
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) throw new Error('Seed must be an unsigned 32-bit integer.');
  if (!Number.isInteger(ticks) || ticks < 1 || ticks > 360) throw new Error('Ticks must be an integer between 1 and 360.');
  const [fanEfficiency, pathRestriction, freezeSensorAt] = FIXTURES[scenarioId];
  const random = rng(seed); const trace = []; let previousReportedAirflow;
  for (let tick = 0; tick < ticks; tick += 1) {
    const ambientVariation = (random() - .5) * .3;
    const targetAirflow = 8.8 + Math.sin((tick + (seed % 17)) / 5) * .55;
    const airflow = Math.max(.1, targetAirflow * fanEfficiency * (1 - pathRestriction) + ambientVariation);
    const reportedAirflow = freezeSensorAt !== undefined && tick >= freezeSensorAt ? previousReportedAirflow : airflow;
    previousReportedAirflow = reportedAirflow;
    trace.push({ tick, scenarioId, temperatureC: round(21.2 + (targetAirflow - airflow) * .36 + Math.cos(tick / 8) * .18), airflow: round(airflow), residual: round(Math.abs(targetAirflow - airflow)), reportedAirflow: round(reportedAirflow), actionAuthority: 'none' });
  }
  return { scenarioId, seed, ticks, trace, digest: await sha256(trace.map((row) => JSON.stringify(row)).join('\n')) };
}
