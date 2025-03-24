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
Polyphonic singing is one of the most widespread forms of music-making. During a performance, singers must constantly adjust their pitch to stay in tune with one another — a complex skill that requires extensive practice. Research has shown that pitch monitoring tools can assist singers in fine-tuning their intonation during a performance [@BerglinPD22_VisualFeedback_JPM]. Specifically, real-time visualizations of the fundamental frequency (F0), which represents the pitch of the singing voice, help singers in assessing their pitch relative to a fixed reference or other voices.
To support the monitoring of polyphonic singing performances, we developed `pytch`, an interactive Python tool with a graphical user interface (GUI) designed to record, process and visualize multiple voices in real time. The GUI displays vocal spectra and estimated F0-trajectories for all singers, as well as the harmonic intervals between them. Additionally, users can adjust visual and algorithmic parameters interactively to accommodate different input devices, microphone signals, singing styles and use cases. Written in Python, `pytch` utilizes the `libf0-realtime` library [@MeierSM25_RealTimeF0_ISMIR] for real-time F0-estimation and the `pyqtgraph` library [^1] to visualize the analysis results.

[^1]: <https://www.pyqtgraph.org>

# Statement of Need
Software that assesses the pitch of a singing voice in real time is mostly known from Karaoke applications, such as Let's Sing[^2] or Rock Band[^3], where the singer sings along a backing track and the pitch is compared to a score reference in order to analyse whether the sung notes are "correct" or "incorrect". However, these applications face several limitations when applied to polyphonic singing. Most notably, many Karaoke systems can only process one or two singing voices at a time, which is problematic for analyzing group performances. Additionally, software that rely on a score as a reference pose challenges for a cappella performances, where singers may drift together in pitch over time while maintaining relative harmony, or in orally-transmitted traditions that may lack a formal score altogether [@ScherbaumMRM19_MultimediaRecordings_FMA]. Finally, existing open-source research software for singing voice processing, like Praat [@Boersma01_Praat_GI], Sonic Visualiser [@CannamLS10_SonicVisualizer_ICMC], and Tarsos [@SixCL13_Tarsos_JNMR], lack real-time feedback, preventing an effective feedback loop between singers and their tool.
To address these challenges, we developed `pytch`. Unlike existing tools, `pytch` enables monitoring multiple voices simultaneously in real time. Rather than classifying singing as simply “correct” or “incorrect”, it serves as an objective monitoring tool for singers and conductors to assess, discuss, and improve their collective tuning. Additionally, `pytch` offers researchers in music information retrieval a platform for developing and testing real-time audio processing algorithms, such as for F0-estimation.

* More explicit use case rehearsal
* more explicit use case tech
* 2 target audiences/users

[^2]: <https://en.wikipedia.org/wiki/Let%27s_Sing>
[^3]: <https://en.wikipedia.org/wiki/Rock_Band_4>

# Audio Processing

In the following, we describe the real-time audio processing pipeline of `pytch`. As shown in Figure X, it consists of two main processes, the recording and the analysis process. The recording process records multichannel audio waveforms from the soundcard or an external audio interface using the `sounddevice` library. The library is based on PortAudio and supports a wide range of operating systems, audio devices, and sampling rates. The recorded audio is received in chunks via a recording callback and fed into a ring buffer shared with the analysis process. When the buffer is sufficiently filled with audio chunks, the analysis process reads the recorded audio to compute several audio features. For each channel, it computes the audio level in dBFS, magnitude short-time fourier transform (also referred to as spectrogram), and an estimate of the F0 using the `libf0-realtime` library [@MeierSM25_RealTimeF0_ISMIR]. The library includes several real-time implementations of well-known F0-estimation algorithms, such as YIN[@CheveigneK02_YIN_JASA] and SWIPE. YIN is a time-domain algorithm that computes the F0 based on a tweaked auto-correlation function. SWIPE is a frequency-domain algorithm that estimates the F0 by matching different spectral representations of the audio with sawtooth-like kernels. The F0-estimates, which are natively computed in the unit Hz are converted to the unit cents using a user-specified reference frequency. Depending on the audio quality and singing voice, estimated F0s may exhibit sudden jumps or incontinuities that result in hard-to read trajectories. To this end, the processing includes to optional filtering operations, one for smoothing using a median filter and one to remove pitch slide artifacts using a gradient filter. As a last audio feature, the harmonic intervals between the F0-trajectories are computed. Every audio feature is stored separately in a dedicated ring buffer. Once the processing is completed, the audio pipeline informs the GUI via a flag that new data for visualization is available.

* refer to basic literature in audio processing
* peter meier paper
* explain confidence here? explain artifacts?


# Graphical User Interface (GUI)

In the following, we explain step-by-step the GUI of `pytch`. Right after the program start, a startup menu opens in which the user is asked to specify the soundcard, the input channels, the sampling rate, and the window size for processing. Furthermore, the user can choose to store the recorded audio and the F0-trajectories on disk. These initial settings are important tfor initialising the audio processing module and the main GUI which is loaded when the user clicks `ok`. While there is no hard limit on the number of channels, we recommend to use up to four input channels to ensure visibility of the charts and responsiveness of the GUI. A screenshot of the main GUI is shown in Figure \autoref{fig:GUI}.

![`pytch` GUI.\label{fig:GUI}](../pictures/screenshot.png){ width=90% }

The GUI is divided in three horizontal parts. The left side contains a menu that includes a start/stop button, and controls to adjust the GUI appearance and tune the algorithmic parameters. In the center of the `pytch` GUI, we find the so called "channel views" - visualisations in a dedicated color for each channel consisting of a level meter for microphone gain control, a spectrum visualisation with the current F0-value marked with a vertical line, and a spectrogram visualisation. The menu on the left can be used to enable or disable views, change the magnitude scaling of the spectrum and the spectrogram, and change the visible frequency range. The channels are ordered from top to bottom in order of selection in the startup menu. Optionally, the bottom channel can show a product of all channels.

The right side of the `pytch` GUI contains the so called "trajectory view" which, depending on the chosen tab, either visualizes the F0-trajectories of all voices ("pitches"), or the harmonic intervals between the voices ("differential"). Using the controls in the left-side menu, the user can select the F0-estimation algorithm and adjust the visualisation using three parameters. First, the user can increase "confidence threshold" parameter to reduce potential noisy F0-estimates. Second, the median smoothing parameter allows to smoothen the displayed F0-trajectories - the larger the filter size the smoother. Thirs, the pitchslide tolerance defines 



# Acknowledgements
**DFG? Uni Potsdam?**
We would like to thank all the singers who contributed to testing `pytch` during its development. The International Audio Laboratories Erlangen are a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS.

# References
