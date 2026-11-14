# aas: guide for agents

`aas` turns text documents into audio files and tells you, in numbers and
words, what the result sounds like. You cannot hear; the measurements are your
ears. Every command prints one JSON object to stdout with `"ok": true` or
`"ok": false` plus a structured error (`code`, `message`, `path`, `hint`).

## The loop

1. **Write** a patch (a sound) or a score (music) as YAML or JSON, or start
   from a built-in: `aas library list`, `aas library show sfx/explosion`.
2. **Render**: `aas render my.yaml -o out.wav`. The response includes an
   `analysis` block (description, loudness, brightness, attack, decay, pitch)
   and, when the document names a `profile`, a `lint` block.
3. **Read** `analysis.description`, `analysis.issues` and `lint.issues`. Each
   issue has a `fix` that says what to change.
4. **Edit and render again** until `lint.passed` is true and the description
   matches what was asked for.
5. To match a reference recording: `aas compare mine.wav reference.wav` and
   follow `how_to_get_closer`.

A human should audition the final assets. Measurements catch technical faults
and gross mismatches; they do not judge taste.

## Commands

| Command | Purpose |
|---|---|
| `aas sfx <recipe> [--set k=v] [-o file]` | Render a built-in sound-effect recipe |
| `aas render <doc> [-o file]` | Render a patch or score |
| `aas validate <doc>` | Check a document without rendering |
| `aas analyze <wav> [--detail] [--loop]` | Full measurements of any WAV |
| `aas describe <wav>` | Description plus a text picture of energy over time |
| `aas lint <wav> --profile <name>` | Check against a use-case profile (exit code 5 on failure) |
| `aas compare <candidate> <reference>` | Similarity score and what to change |
| `aas spectrogram <wav> -o img.png` | Image, if you can view images |
| `aas library list / show / copy` | Browse and copy built-in recipes, instruments, drums, examples |
| `aas nodes [name]` / `aas nodes --functions` | Reference for node types and expressions |
| `aas schema [patch|score|pack]` | Every key of each document type |
| `aas profiles` | Lint profiles |
| `aas theory key|chord|note|scales` | Notes of a key or chord |
| `aas pack <manifest>` | Render many assets and write `manifest.json` |

Useful render flags: `--set name=value` (patch parameters), `--seed N`,
`--variations N` (round-robin variants), `--loop` (seamless loop),
`--stems` and `--sections` (scores), `--quality draft|standard|high`,
`--probe` (level of every node: find the layer that is too loud or silent),
`--raw` (no normalization), `--profile name`, `--sr`, `--bits 16|24|32f`.

Exit codes: 0 ok, 2 usage, 3 invalid document, 4 file error, 5 lint failed, 1 other.

## Patches: one sound

A patch is a graph of named nodes. Each node has a `type` and parameters.
Any numeric parameter can be a number or a quoted expression. Expressions can
use other node names (their output signals), patch `params`, and the built-in
variables `t` (seconds), `dur`, `freq`, `vel`, `gate`.

```yaml
description: Laser shot
duration: 0.5                 # seconds; may be an expression of params
profile: weapon               # lint profile checked on every render
params:
  pitch: {default: 2600, min: 600, max: 6000, doc: "start pitch in Hz"}
nodes:
  env:  {type: env, points: [[0, 1], [0.3, 0]], curve: exp}   # loudness shape
  f:    {type: sweep, from: pitch, to: 160, time: 0.22}       # falling pitch
  osc:  {type: osc, wave: saw, freq: f, amp: env}
  filt: {type: lowpass, in: osc, cutoff: "min(f * 4, 16000)", q: 3}
  out:  {type: reverb, in: filt, decay: 0.5, mix: 0.12}       # node named `out` is the output
```

Rules that prevent most mistakes:

- Quote every expression: `cutoff: "200 + 4000 * env"`.
- Shape loudness with an envelope: `amp: env` on a source, or `"sig * env"`.
- Sounds must end in silence before `duration` (or be loops). Reverb and delay
  tails need extra duration.
- `rand(lo, hi)` gives one random constant per render; it changes with
  `--seed`, which is how `--variations` produces different takes.
- Output is normalized to -1 dBFS peak by default. Use
  `normalize: {lufs: -20}` for loudness-matched sets or `normalize: none`.
- `loop: true` (or `--loop`) crossfades the end into the start.

Building blocks (run `aas nodes` for all, `aas nodes <name>` for parameters):

- Sources: `osc` (sine, saw, square, pulse, triangle; `unison` for width;
  `pm` for FM), `noise`, `dust` (random clicks), `impulse`, `additive`,
  `pluck` (string), `modal` (struck metal, wood, glass, membranes).
- Modulators: `env` (breakpoints), `adsr` (gate-driven), `lfo`, `sweep`.
- Filters: `lowpass`, `highpass`, `bandpass`, `notch`, `peak`, `lowshelf`,
  `highshelf`, `comb`, `formant` (vowels), `dcblock`.
- Shapers and dynamics: `distort`, `bitcrush`, `compressor`, `limiter`,
  `transient`.
- Space: `reverb`, `delay`, `chorus`, `flanger`, `phaser`, `pan`, `width`.
- Utility: `mix`, `expr`, `gain`, `fade`, `shift`, `reverse`, `speed`.

### Sound-design recipes in words

- **Impact**: a click or noise burst at t=0 (1-5 ms), a body (pitched sine
  dropping in frequency, or `modal`), an optional tail (filtered noise,
  reverb). Add `transient` with `attack_db: 6` for punch.
- **Whoosh**: noise through a `bandpass` whose cutoff rises then falls, with
  a loudness envelope that swells and fades; `pan` moving across.
- **Laser / zap**: oscillator with a fast downward `sweep`, short envelope.
- **Explosion**: crack + lowpassed noise with falling cutoff + sub sine drop
  + rumble; `distort` for grit; `reverb` for distance.
- **Magic / sparkle**: `dust` into `modal` (glass, bowl) at high pitches,
  with `chorus` and `reverb`.
- **Ambience loop**: stereo noise layers through filters whose cutoffs follow
  slow `lfo` nodes (wave: smooth); `loop: {crossfade: 2}`.
- **Creature**: saw with jittered pitch through `formant`, then `distort`.
- **UI**: one or two short sine/triangle notes, 30-300 ms, little bass.
- Darker = lower `lowpass` cutoff. Further away = darker, more reverb, softer
  attack. Bigger = lower pitch, longer decay. More aggressive = more `distort`.

## Scores: music

```yaml
kind: score
tempo: 100
key: C minor                       # enables out-of-key warnings
instruments:
  drums: {kit: {kick: builtin:drums/kick, snare: builtin:drums/snare, hat: builtin:drums/hat_closed}}
  bass:  {patch: builtin:instruments/bass_synth}
  pad:   {patch: builtin:instruments/pad_warm, sends: {hall: -8}}
buses:
  hall: {fx: [{type: reverb, decay: 2.5, mix: 1}]}
patterns:
  beat:     {drums: {kick: "x...x...x...x...", snare: "....x.......x...", hat: "x.x.x.x.x.x.x.x."}}
  bassline: {notes: "C2:1 C2:0.5 Eb2:0.5 G2:1 Bb1:1 | Ab1:1 Ab1:0.5 C2:0.5 Eb2:1 G1:1"}
  chords:   {chords: "Cm:4 Ab:4"}
sections:
  main: {bars: 4, tracks: {drums: beat, bass: bassline, pad: chords}}
arrangement: [main, main]
```

- **Notes**: `NOTE:BEATS@VELOCITY`, separated by spaces. `C4` is middle C.
  `_:1` is a rest, `~:1` extends the previous note, `C4+E4+G4:2` is a chord,
  `|` is a bar line (bars are checked against the time signature, which
  catches rhythm mistakes). A list of strings is one string per bar.
- **Drums**: one step string per kit piece. `x` hit, `X` accent, `o` ghost,
  `.` rest. Each character is a sixteenth note unless `step` says otherwise.
- **Chords**: `SYMBOL:BEATS`. `style` turns them into `arp_up`, `arp_down`,
  `arp_updown`, `strum` or `root` (bass notes). `voicing: smooth` keeps
  movement between chords small.
- **Sections** assign patterns to instruments. A list of patterns plays in
  turn and repeats to fill the section.
- **Mixing**: per-instrument `gain_db`, `pan`, `fx` (list of effect nodes),
  `sends` to buses, and `duck` (sidechain: `duck: {source: drums.kick,
  amount_db: 4}`). `master: {lufs: -16}` sets the final loudness; a true-peak
  limiter holds -1 dBFS.
- **Deliverables**: `--loop` for a seamless loop (the tail wraps into the
  start), `--stems` for one file per instrument, `--sections` for one file
  per section with its tail, for adaptive music.
- `aas theory key "D minor"` lists the notes and chords that fit a key.

Composition guidance that the measurements cannot give you:

- Keep melodies mostly stepwise with a few leaps; end phrases on chord tones.
- Bass plays chord roots on strong beats; keep it below C3.
- Leave space: not every instrument in every section. Build by adding layers.
- Typical tempos: ambient 60-80, calm 80-100, action 120-150, chiptune 140-180.
- Check the result: `analysis.tempo_bpm` and `analysis.key` should match what
  you wrote; `--stems` plus `aas analyze` shows which instrument dominates.

## Reading the measurements

| Field | Meaning | Typical use |
|---|---|---|
| `lufs` | Perceived loudness | Match assets of one category to each other |
| `true_peak_db` | Highest level incl. between samples | Must stay at or below -1 |
| `centroid_hz` | Brightness | <800 dark, 800-2500 balanced, >2500 bright |
| `attack_ms` | Time to reach full level | Impacts and UI need <10-20 ms |
| `decay_s` | Time to fall 20 dB after the peak | Length of the tail |
| `bands` | Share of energy per frequency band | Find missing or excessive bass/treble |
| `pitch` | Detected note, and start/end for sweeps | Confirm tuning and pitch movement |
| `stereo.width` | 0 mono .. 1 very wide | Ambiences want width, UI does not need it |
| `loop_seam` | Discontinuity at the loop point | Must not be `audible` |
| `issues` | Concrete faults with fixes | Fix all errors, consider warnings |

`aas describe file.wav` adds `energy_map`: rows are frequency bands, columns
are time; denser characters mean more energy. Use it to see when each part of
the spectrum is active.

## Profiles

`ui`, `oneshot`, `impact`, `weapon`, `whoosh`, `ambience-loop`, `sfx-loop`,
`music-loop`, `music`, `stinger`. Set `profile:` in a patch, or pass
`--profile` to `render` or `lint`. See `aas profiles` for what each checks.

## Limits

- Synthesis only. Realistic acoustic instruments, speech and recorded foley
  are out of reach; the built-in "strings", "brass" and "choir" are synth
  renditions.
- Scores have one tempo and time signature, and notes do not glide.
- Stems are held in memory: very long scores with `--stems` need several GB.
