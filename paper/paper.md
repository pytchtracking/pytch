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
date: 11 April 2025
bibliography: paper.bib
---

# Summary
Polyphonic singing is one of the most widespread forms of music-making. During a performance, singers must constantly adjust their pitch to stay in tune with one another — a complex skill that requires extensive practice. Research has shown that pitch monitoring tools can assist singers in fine-tuning their intonation during a performance [@BerglinPD22_VisualFeedback_JPM]. Specifically, real-time visualizations of the fundamental frequency (F0), which represents the pitch of the singing voice, help singers assess their pitch relative to a fixed reference or other voices.
To support the monitoring of polyphonic singing performances, we developed `pytch`, an interactive Python tool with a graphical user interface (GUI) designed to record, process, and visualize multiple voices in real time. The GUI displays vocal spectra and estimated F0 trajectories for all singers, as well as the harmonic intervals between them. Additionally, users can adjust visual and algorithmic parameters interactively to accommodate different input devices, microphone signals, singing styles and use cases. Written in Python, `pytch` utilizes the `libf0-realtime` library [@MeierSM25_RealTimeF0_ISMIR] for real-time F0 estimation and `pyqtgraph`[^1] for efficient visualizations of the analysis results.

[^1]: <https://www.pyqtgraph.org>

# Statement of Need
Software that assesses the pitch of a singing voice in real time is mostly known from Karaoke singing applications, such as Let's Sing[^2], Rock Band[^3], or Cantamus[^4]. These tools typically compare the singer’s pitch to a score reference to judge whether notes are ‘correct’ or ‘incorrect’. However, such applications face several limitations when applied to polyphonic or group singing contexts. Most notably, many Karaoke systems can only process one or two singing voices at a time, which is problematic for monitoring group performances. Additionally, software that relies on a score as a reference poses challenges for a cappella performances, where singers may drift together in pitch over time while maintaining relative harmony, or in orally-transmitted traditions that may lack a formal score altogether [@ScherbaumMRM19_MultimediaRecordings_FMA]. Finally, existing open-source research software for singing voice processing, like Praat [@Boersma01_Praat_GI], Sonic Visualiser [@CannamLS10_SonicVisualizer_ICMC], and Tarsos [@SixCL13_Tarsos_JNMR], lack real-time feedback, preventing an effective feedback loop between singers and their tool.

To address these challenges, we developed `pytch`, a tool that enables singers and conductors to analyze and improve various aspects of a performance during rehearsals. For example, the vocal spectra can help singers fine-tune the expression of formant frequencies, while melodic and harmonic issues become visible through F0 trajectories and harmonic intervals. Unlike many existing tools, `pytch` does not require a musical score, making it well-suited for a wide variety of singing traditions, including a cappella and orally transmitted genres.

In addition to its practical applications, `pytch` also provides a flexible platform for music information retrieval (MIR) research on real-time audio processing. Working with real-time data introduces challenges such as a limited audio context for analysis and strict timing constraints to ensure low-latency processing. Researchers can use `pytch` to develop, test, and compare algorithms for tasks like F0 estimation and signal enhancement [@MeierCM24_RealTimePLP_TISMIR].

[^2]: <https://en.wikipedia.org/wiki/Let%27s_Sing>
[^3]: <https://en.wikipedia.org/wiki/Rock_Band_4>
[^4]: <https://cantamus.app>

# Audio Processing

The real-time audio processing pipeline implemented in the file `audio.py` is the heart of `pytch` and consists of two main stages: recording and analysis. The recording stage records multichannel audio waveforms from the soundcard or an external audio interface using the `sounddevice` library. The library is based on PortAudio and supports a wide range of operating systems, audio devices, and sampling rates. The recorded audio is received in chunks via a recording callback and fed into a ring buffer shared with the analysis process. When the buffer is sufficiently filled with audio chunks, the analysis process reads the recorded audio to compute several audio features.

For each channel, the analysis stage computes the audio level in dBFS, a time-frequency representation of the audio signal via the Short-Time Fourier Transform (see [@Mueller21_FMP_SPRINGER] for fundamentals of music processing), and an estimate of the F0 along with a confidence value using the `libf0-realtime` library [@MeierSM25_RealTimeF0_ISMIR]. The library includes several real-time implementations of well-known F0 estimation algorithms, such as YIN [@CheveigneK02_YIN_JASA] and SWIPE. YIN is a time-domain algorithm that computes the F0 based on a tweaked auto-correlation function. It is computationally efficient and well-suited for low-latency applications, but it tends to suffer from estimation errors, particularly confusions with higher harmonics such as the octave, for noisy signals. In contrast, SWIPE is a frequency-domain algorithm that estimates the F0 by matching different spectral representations of the audio with sawtooth-like kernels. While more computationally demanding, SWIPE typically yields more reliable estimates, in particular for vocal input signals. `pytch` allows users to choose between these algorithms depending on their specific needs and system capabilities. The obtained F0 estimates, which are natively computed in the unit Hz are converted to the unit cents using a user-specified reference frequency. Depending on the audio quality and vocal characteristics, F0 estimates may exhibit artifacts such as discontinuities or pitch slides, which can make the resulting trajectories difficult to interpret [@RosenzweigSM19_StableF0_ISMIR]. Previous research has shown that using throat microphones can improve the isolation of individual voices in group singing contexts, resulting in cleaner signals and more accurate F0 estimates [@Scherbaum16_LarynxMicrophones_IWFMA]. To further enhance interpretability, `pytch` includes several optional post-processing steps: a confidence threshold to discard estimates with low confidence score, a median filter to smooth the trajectories, and a gradient filter to suppress abrupt pitch slides. As a final step in the audio analysis, the harmonic intervals between the F0 trajectories are computed. Every audio feature is stored separately in a dedicated ring buffer. After processing, the pipeline sets a flag that notifies the GUI that new data is ready for visualization.


# Graphical User Interface (GUI)

In this section, we provide a step-by-step explanation of the `pytch` GUI implemented in the file `gui.py`. Right after the program start, a startup menu opens in which the user is asked to specify the soundcard, input channels, sampling rate, and window size for processing. Furthermore, the user can choose to store the recorded audio and the F0 trajectories on disk. These configuration choices are required to initialize the audio processing module and the main GUI which is loaded when the user clicks "ok". While there is no hard limit on the number of channels, we recommend to use up to four input channels to ensure visibility of the charts and responsiveness of the GUI. A screenshot of the main GUI is shown in Figure \autoref{fig:GUI}.

![`pytch` GUI monitoring three singing voices.\label{fig:GUI}](../pictures/screenshot.png){ width=90% }

The main GUI is organized into three horizontal sections. On the left, a control panel provides a start/stop button and allows users to adjust both the visual layout and algorithmic parameters. The central section displays "channel views"--one for each input channel--color-coded for clarity. Each view includes a microphone level meter, a real-time spectrum display with a vertical line marking the current F0 estimate, and a scrolling spectrogram with a 5 second time context. Channels are listed from top to bottom in the order they were selected during setup. Optionally, the bottommost view can display a product signal from all channels.

The right section, referred to as the "trajectory view," provides time-based visualizations of either the F0 trajectories ("pitches" tab) or the harmonic intervals between voices ("differential" tab) with a 10 second time context. Using the controls in the left-side menu, the user can select the F0 estimation algorithm and improve the real-time visualization by adjusting the confidence threshold, the median filter length for smoothing and the tolerance of the gradient filter. F0 and interval trajectories can be displayed with respect to a fixed reference frequency or a dynamic one derived from a selected channel, the lowest, or highest detected voice. Axis limits for this section can also be manually set.

# Acknowledgements
**DFG? Uni Potsdam?**
We would like to thank all the singers who contributed to testing `pytch` during its development. The International Audio Laboratories Erlangen are a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS.

# References
