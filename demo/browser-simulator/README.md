# Browser-local deterministic simulator

Open `index.html` directly in a modern browser; it performs no network requests.
It recreates the accepted browser-simulator behavior as four closed deterministic
fixtures. It is deliberately separate from the Python forecast receipt:

- **browser simulator:** local fixture exploration only; `actionAuthority: "none"`;
- **judge command:** runs the bounded Python forecast, then independently verifies
a deterministic HMC replay; the forecast remains advisory-only and HMC is the
sole command authority.

This is not a hardware controller, validation, certification, qualification, or
performance claim.

```bash
node --test demo/browser-simulator/simulator.test.mjs
```
