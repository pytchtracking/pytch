---
title: '`pytch`: A Real-Time Monitoring Tool For Polyphonic Singing Performances'
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
    affiliation: 2
  - name: Meinard Müller
    orcid: 0000-0001-6062-7524
    corresponding: false
    affiliation: 2
  - name: Frank Scherbaum
    orcid: 0000-0002-5050-7331
    corresponding: false
    affiliation: 3


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
Polyphonic singing is one of the most widespread forms of music-making. During a performance, singers must constantly adjust their pitch to stay in tune with one another — a complex skill that requires extensive practice. Research has shown that tools offering visual feedback can assist singers in fine-tuning their intonation during a performance [@BerglinPD22_VisualFeedback_JPM]. Specifically, real-time visualizations of the fundamental frequency (F0), which represents the pitch of the singing voice, help singers in assessing their pitch relative to a fixed reference or other voices.
To support the monitoring of polyphonic singing performances, we developed `pytch`, an interactive Python tool with a graphical user interface (GUI) designed to record, process and visualize multiple voices in real time. The GUI displays vocal spectra and estimated F0-trajectories for all singers, as well as the harmonic intervals between them. Additionally, users can adjust visual and algorithmic parameters interactively to accommodate different input devices, microphone signals, singing styles and use cases. Written in Python, `pytch` utilizes the `libf0-realtime` library [@MeierSM25_RealTimeF0_ISMIR] for real-time F0-estimation and the `pyqtgraph` library [^1] to visualize the analysis results.

[^1]: <https://www.pyqtgraph.org>

# Statement of Need
Software for assessing a singing voice performance
Various tools for singing performance analysis exist, ranging from open-source research platforms like Praat [@Boersma01_Praat_GI], Sonic Visualiser [@CannamLS10_SonicVisualizer_ICMC], and Tarsos [@SixCL13_Tarsos_JNMR], to commercial applications such as Singstar[^2] or Singing Carrots [^3]. However, these tools face several limitations when applied to polyphonic singing. Most notably, many tools can only process a single voice at a time, which is problematic for analyzing the interactions between voices in a group performance. Additionally, real-time feedback is often missing, preventing an effective feedback loop between singers and their tool. Furthermore, tools that rely on a score as a reference pose challenges for a cappella performances, where singers may drift together in pitch over time while maintaining relative harmony, or in orally-transmitted traditions that may lack a formal score altogether [@ScherbaumMRM19_MultimediaRecordings_FMA]. To address these challenges, we developed `pytch`. Unlike existing tools, `pytch` enables monitoring multiple voices simultaneously in real time


without classifying singing as simply “correct” or “incorrect.” Instead, it serves as an objective, score-independent measurement tool for singers and conductors to assess and improve their collective tuning. Additionally, `pytch` offers a platform for developing and testing real-time audio processing algorithms, such as for F0-estimation.

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
We would like to thank all the singers who contributed to testing `pytch` during its development. The International Audio Laboratories Erlangen are a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS.

# References
