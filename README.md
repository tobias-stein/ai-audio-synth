# aas - AI audio synth

`aas` is a command-line tool that lets an AI agent create sound effects and
music for games without being able to hear. The agent writes a text document
describing a sound or a piece of music, `aas` renders it to a WAV file with
its own synthesis engine, and reports back measurements, a plain-language
description, a list of concrete problems and how to fix each one.

```
write YAML  ->  aas render  ->  read the JSON report  ->  edit  ->  repeat
```

Everything is procedural: there are no samples and no network calls. The same
document and seed always produce the same file, bit for bit.

## Build

Requires a recent stable Rust toolchain (developed with Rust 1.97).

```sh
cargo build --release      # binary at target/release/aas
cargo test --release       # unit tests plus end-to-end tests of the whole library
```

## Quick start

```sh
aas guide                                   # the full guide for agents (Markdown)
aas library list --category sfx             # built-in sound-effect recipes
aas sfx explosion --set size=0.9 -o boom.wav
aas sfx sword_swing --variations 4 -o sfx/  # four round-robin takes
aas sfx wind -o wind.wav                    # a seamless loop with embedded loop points
aas render builtin:examples/battle_theme -o battle.wav --stems --sections
aas analyze boom.wav                        # measure any WAV
aas lint boom.wav --profile impact          # check it against its intended use
aas compare mine.wav reference.wav          # how to get closer to a reference
```

Every command prints one JSON object. A render reports what it made (abridged):

```json
{
  "ok": true,
  "outputs": [{
    "path": "boom.wav",
    "duration_s": 4.925,
    "analysis": {
      "lufs": -14.6,
      "true_peak_db": -1.0,
      "centroid_hz": 1467.0,
      "attack_ms": 0.4,
      "decay_s": 0.9,
      "description": "Long (4.92 s), balanced, inharmonic sound with a sharp, percussive attack and a long tail (0.9 s to fall 20 dB). Most energy (50%) is in the bass (60-250 Hz). Wide stereo image. -14.6 LUFS with punchy dynamics.",
      "issues": []
    },
    "lint": { "profile": "impact", "passed": true, "errors": 0, "warnings": 0, "issues": [] }
  }]
}
```

Errors are structured too, with the location and a suggested fix:

```json
{ "ok": false, "error": {
    "code": "invalid_document",
    "message": "node type 'osc' has no parameter 'frq'",
    "path": "nodes.a.frq",
    "hint": "did you mean 'freq'? Valid: wave, freq, amp, pw, pm, unison, detune, spread, phase" } }
```

## Documents

**Patch** - one sound, as a graph of nodes. Any number can be an expression of
other nodes, parameters and time.

```yaml
description: Laser shot
duration: 0.5
profile: weapon
params:
  pitch: {default: 2600, min: 600, max: 6000}
nodes:
  env:  {type: env, points: [[0, 1], [0.3, 0]], curve: exp}
  f:    {type: sweep, from: pitch, to: 160, time: 0.22}
  osc:  {type: osc, wave: saw, freq: f, amp: env}
  filt: {type: lowpass, in: osc, cutoff: "min(f * 4, 16000)", q: 3}
  out:  {type: reverb, in: filt, decay: 0.5, mix: 0.12}
```

**Score** - music: instruments, patterns, sections and an arrangement, with a
mixer (gain, pan, effects, send buses, sidechain ducking) and mastering to a
loudness target.

```yaml
tempo: 100
key: C minor
instruments:
  drums: {kit: {kick: builtin:drums/kick, snare: builtin:drums/snare, hat: builtin:drums/hat_closed}}
  bass:  {patch: builtin:instruments/bass_synth}
patterns:
  beat:     {drums: {kick: "x...x...x...x...", snare: "....x.......x...", hat: "x.x.x.x.x.x.x.x."}}
  bassline: {notes: "C2:1 C2:0.5 Eb2:0.5 G2:1 Bb1:1"}
sections:
  main: {bars: 4, tracks: {drums: beat, bass: bassline}}
```

**Pack** - a batch of assets rendered in one go, with a `manifest.json`.

`aas schema` lists every key; `aas nodes` documents the 40 node types.

## What is in the box

- **Engine**: band-limited oscillators, FM, additive, noise, plucked-string and
  modal (struck object) models; state-variable filters and EQ; distortion with
  internal oversampling; compressor, transient shaper and true-peak limiter;
  delay, chorus, flanger, phaser and a modulated feedback-delay-network reverb.
  64-bit float throughout, 2x or 4x oversampled, dithered 16/24-bit or float output.
- **Library**: 36 sound-effect recipes (UI, pickups, weapons, impacts, motion,
  magic, sci-fi, creatures, looping ambiences), 19 instruments, 17 drums and
  6 example scores. All are plain YAML: `aas library show <id>`.
- **Measurement**: loudness (ITU-R BS.1770), true peak, spectrum and band
  balance, envelope, attack and decay, pitch and pitch movement, onsets, tempo,
  key, stereo width and mono compatibility, loop-seam continuity.
- **Game deliverables**: seeded variations, seamless loops with `smpl` loop
  points, stems, per-section exports with ring-out tails for adaptive music.

## Quality and limits

- The engine is built to professional technical standards (no aliasing from
  oscillators or distortion, true-peak-safe output, loudness-normalized
  masters) and every built-in asset is checked by the test suite against its
  lint profile.
- Measurements verify technical correctness and catch gross mismatches. They
  do not judge taste. **Have a person audition assets before shipping them.**
- Synthesis suits UI, sci-fi, magic, impacts, whooshes, ambiences, drones and
  electronic or hybrid music. It does not replace recorded foley, voices or a
  sampled orchestra; the "strings", "brass" and "choir" instruments are synth
  renditions.
- Scores have a single tempo and time signature, and notes do not glide.

## Layout

```
src/dsp/        oscillators, filters, effects, reverb, dynamics, resampling
src/graph.rs    patch parsing, validation and graph evaluation
src/nodes.rs    node registry (drives validation and `aas nodes`) and implementations
src/expr.rs     expression language
src/score.rs    score parsing, sequencer, mixer, mastering
src/theory.rs   keys, chords, note and drum notation
src/analysis/   measurements, lint profiles, comparison, spectrogram
src/cli.rs      commands and JSON output
library/        built-in recipes, instruments, drums and examples (embedded at build time)
docs/           the agent guide printed by `aas guide`
tests/cli.rs    end-to-end tests
```

## License

MIT
