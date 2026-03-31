# Axial Force-Moment-Curvature Calculation Model for SFCB Reinforced UHPC Rectangular Sections

Developed on the Python platform, the "Axial Force-Moment-Curvature Calculation Model for SFCB Reinforced UHPC Rectangular Sections" integrates a full-process cross-sectional mechanical analysis algorithm based on the numerical iteration method.

## Calculation Principle
The calculation applies curvature increments step by step. At a given curvature, the compressive edge concrete strain is initially assumed. Based on the plane section assumption, the strain of each fiber strip is determined. Combined with the built-in constitutive relationships of UHPC (or Normal Strength Concrete) and SFCB, the stress distribution is solved. Subsequently, the internal forces of all strips are integrated to check the cross-sectional force equilibrium condition. If equilibrium is not satisfied, the bisection method is used to modify the assumed strain until convergence. Once equilibrium is achieved, the corresponding internal moment is calculated. This process is repeated until the predefined ultimate state is reached, thereby outputting precise full-process analysis results of the member's mechanical behavior.

## Included Files
- `GUI.py`: Main program with Graphical User Interface (Python source code).
- `GUI.exe`: Standalone executable file ready to run (Windows).
- `GUI_Code.txt`: Plain text code for easy copy-pasting into Word documents.
- `MNPHI.m`: The original MATLAB algorithm file.
- `MNPHI.py`: The basic Python translation of the MATLAB algorithm.

## Usage Instructions
1. Double-click to run `GUI.exe`.
2. Input the cross-sectional geometric parameters, concrete parameters (supports switching between UHPC and NSC), and double-row reinforcement parameters in the top panel.
3. Click "Start Calculation" to begin the analysis.
4. Once the calculation is complete, the Moment-Curvature curve will be automatically plotted in the bottom-left area.
5. Check the desired data indicators in the bottom-right panel, and click "Export to Excel" to export the full-process data.