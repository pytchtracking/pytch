---
title: '`pytch`: A Real-Time Analysis Tool For Polyphonic Singing'
tags:
  - Python
  - Audio Processing
  - Music Information Retrieval
  - Singing Voice Analysis
  - GUI
authors:
  - name: Sebastian Rosenzweig
    orcid: 0000-0003-4964-9217
    corresponding: true
    affiliation: "1,2"
  - name: Marius Kriegerowski
    orcid:
    corresponding: false
    affiliation: 4
  - name: Lukas Dietz
    orcid:
    corresponding: false
    affiliation: "1,5"
  - name: Frank Scherbaum
    orcid: 0000-0002-5050-7331
    corresponding: false
    affiliation: 3
  - name: Meinard Müller
    orcid: 0000-0001-6062-7524
    corresponding: false
    affiliation: 2


affiliations:
 - name: Audoo Ltd., London, United Kingdom
   index: 1
 - name: International Audio Laboratories Erlangen, Erlangen, Germany
   index: 2
 - name: University of Potsdam, Potsdam, Germany
   index: 3
 - name: Independent Researcher, Berlin, Germany
   index: 4
 - name: Tantive GmbH, Nürnberg, Germany
   index: 5
date: 7 March 2025
bibliography: paper.bib
---

# Summary
Polyphonic singing is one of the most widespread forms of music-making. During a vocal performance, singers must constantly adjust their pitch to stay in tune with one another — a complex skill that often requires extensive practice and guidance from a conductor or experienced lead singer. Recent research suggests that a singer can improve their tuning during rehearsals when given visual feedback in real time [@BerglinPD22_VisualFeedback_JPM], e.g., through watching a visualization of the fundamental frequency (F0) trajectory which corresponds to the pitch progression of the singing voice. In the context of polyphonic singing, real-time analysis of all voices together is essential to assess the complex interactions and provide meaningful feedback. To this end, we developed `pytch`, an interactive Python tool with a graphical user interface (GUI) designed to record and analyze multiple voices in real time through multichannel processing. The tool displays vocal spectra and estimated F0-trajectories for all singers, as well as the harmonic intervals between them. Furthermore, the user can interactively tune visual and algorithmic parameters to adapt to different input devices, microphone signals, singing styles, and use cases. Written in Python, `pytch` utilizes the `libf0` library [@RosenzweigSM22_libf0_ISMIR-LBD] for F0-estimation and the `pyqtgraph` library [^1] to visualize the analysis results.

[^1]: <https://www.pyqtgraph.org>

# Statement of Need
Various tools for singing analysis exist, ranging from open-source research platforms like Praat [@Boersma01_Praat_GI], Sonic Visualiser [@CannamLS10_SonicVisualizer_ICMC], and Tarsos [@SixCL13_Tarsos_JNMR], to commercial applications such as Singstar[^2] or Singing Carrots [^3]. However, these tools face several limitations when applied to polyphonic singing. Most notably, many tools can only process a single voice at a time, which is problematic for analyzing the interactions between voices in a group performance. Additionally, real-time feedback is often missing, preventing an effective feedback loop between singers and their tool. Furthermore, tools that rely on a score as a reference pose challenges for a cappella performances, where singers may drift together in pitch over time while maintaining relative harmony, or in orally-transmitted traditions that may lack a formal score altogether. To address these challenges, we developed `pytch`. Unlike existing tools, `pytch` enables the analysis of multiple voices simultaneously in real time without classifying singing as simply “correct” or “incorrect.” Instead, it serves as an objective, score-independent measurement tool for singers and conductors to assess and improve their collective tuning. Additionally, `pytch` offers a platform for developing and testing real-time audio processing algorithms, such as for F0-estimation.

[^2]: <https://de.wikipedia.org/wiki/SingStar>
[^3]: <https://singingcarrots.com/pitch-monitor>

# Audio Processing
*
YIN [@CheveigneK02_YIN_JASA]

# Graphical User Interface (GUI)
![`pytch` GUI.\label{fig:GUI}](../pictures/screenshot.png){ width=90% }

A screenshot of the main GUI is shown in Figure \autoref{fig:GUI}.

While there is no hard limit on the number of channels, we recommend to use up to four input channels to ensure visibility of the charts and responsiveness of the GUI.



# Acknowledgements
**DFG? Uni Potsdam?**
We would like to thank all singers that helped developing this tool. The International Audio Laboratories Erlangen are a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS.

# References
