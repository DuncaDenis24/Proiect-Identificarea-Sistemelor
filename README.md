# Documentation for System Identification Project

## Overview
This project involves identifying and analyzing second-order systems through experimental data, using methods such as non-parametric and parametric identification. It also includes Bode diagram estimation and validation using ARX and ARMAX models.

## Table of Contents
1. [Experimental Data](#1-experimental-data)
2. [Non-Parametric Identification](#2-non-parametric-identification)
   - [Second-Order System Identification](#21-second-order-system-identification)
3. [Bode Diagram Estimation](#3-bode-diagram-estimation)
4. [Parametric Identification](#4-parametric-identification)
   - [ARX Model](#41-arx-model)
   - [ARMAX Model](#42-armax-model)
5. [Second-Order System with Zero](#5-second-order-system-with-zero)
   - [ARX Model](#51-arx-model)
   - [ARMAX Model](#52-armax-model)

---

## 1. Experimental Data
- **Inputs**: `u(y)` represents the system input.
- **Outputs**:
  - `y1(t)`: Output of the second-order system.
  - `y2(t)`: Output of the second-order system with a zero.

---

## 2. Non-Parametric Identification
### 2.1 Second-Order System Identification
**Steps:**
1. **Data Acquisition**:
   - Determine proportionality factor `K` and resonance parameters `Mr` and `ζ`.
2. **Resonance Period** (`Tr`) and Natural Frequency (`ωn`):
   - Formulae:
     ```
     Tr = t(274) - t(257)  
     ωn = ωr / √(1 - 2ζ²)
     ```
3. **Transfer Function**:
   - Example:
     ```
     H(s) = (ωn²) / (s² + 2ζωns + ωn²)
     ```
4. **Simulation and Error Calculation**:
   - Example Error: `4.31%`.

---

## 3. Bode Diagram Estimation
**Steps:**
1. Select points at low, high, and resonant frequencies.
2. Extract frequency (`ω`), magnitude (`M`), and phase (`Φ`).
3. Plot and overlay points with the Bode diagram of the system.
4. Calculate attenuation slope per decade:
   ```
   p = 20 * log10(M24 / M10) / log10(ω24 / ω10)
   ```
   Example: `-44.4464 dB/dec`.

---

## 4. Parametric Identification
### 4.1 ARX Model
**Steps:**
1. Determine acquisition period: `dt = t(2) - t(1)`.
2. Prepare data using the `iddata` function in MATLAB.
3. Apply the ARX function with polynomial degrees `nA = 2, nB = 2, nd = 0`.
4. Validate:
   - Fit to estimation data: `94.34%`.
   - Example error: `5.52%`.

### 4.2 ARMAX Model
**Steps:**
1. Use MATLAB’s ARMAX function with degrees `nA = 2, nB = 2, nC = 2, nd = 1`.
2. Validate:
   - Fit to estimation data: `96.45%`.
   - Example error: `4.1%`.

---

## 5. Second-Order System with Zero
### 5.1 ARX Model
**Steps:**
1. Prepare data using `iddata`.
2. Apply ARX function with polynomial degrees `nA = 3, nB = 2, nd = 0`.
3. Validate:
   - Fit to estimation data: `95.72%`.
   - Example error: `4.08%`.

### 5.2 ARMAX Model
**Steps:**
1. Use MATLAB’s ARMAX function with degrees `nA = 2, nB = 3, nC = 2, nd = 0`.
2. Validate:
   - Fit to estimation data: `96.62%`.
   - Example error: `3.57%`.

---

## Tools and Technologies
- MATLAB for data analysis, model creation, and validation.
- Bode diagram analysis for system behavior visualization.



