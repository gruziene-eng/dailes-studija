# 🎨 Dailės studija

Lokali platforma: iš jūsų iCloud / Photos bibliotekos (arba bet kurio aplanko) atrenka geriausias kelionių
nuotraukas ir kiekvienai sudaro tapybos ant drobės planą. Niekas nesiunčiama į internetą.

## Paleidimas
1. Dukart spustelėkite **`Paleisti studiją.command`** (pirmą kartą macOS gali paklausti – Dešinys pelės klavišas → Atidaryti).
2. Pirmą kartą per 1–3 min. įsidiegs bibliotekos, tada naršyklėje atsidarys http://localhost:8765
3. Norint skaityti Photos biblioteką: **Sistemos nustatymai → Privatumas ir sauga → Visos disko prieigos teisė → ＋ Terminal**.

## Kaip veikia
| Skiltis | Ką daro |
|---|---|
| 1. Nuotraukos | Šaltinis (Photos arba aplankas), kelionių atpažinimas (GPS + datos / albumai / poaplankiai) |
| 2. Kelionės | Kiekvienos kelionės geriausi kadrai su balu 0–100, dublikatai atmesti, ★ – noriu nutapyti |
| 3. Tapybos planas | Drobės dydžiai, paletė su dažų maišymo receptais, tinklelis, kontūrai, 3 ir 5 tonų studijos, spalvų zonos, kompozicija, žingsniai, technikos, pirkiniai |
| 4. Mano drobės | Visi pasirinkti paveikslai ir **bendras** pirkinių sąrašas (akrilas / aliejus) |

**Balas** = tapybiškumas (spalvingumas, paprastumas, gamta: dangus, vanduo, augmenija) + techninė kokybė
(aštrumas, ekspozicija, kontrastas) + Apple estetinis įvertinimas ir „mėgstama“ žymė. Asmenukės, žmonių pilni
kadrai, dokumentai ir ekrano nuotraukos nustumiami žemyn.

## Duomenys
Talpykla (miniatiūros, rezultatai): `~/Library/Application Support/DailesStudija` – galima saugiai ištrinti.
Python aplinka: `~/.venvs/dailes-studija`.
