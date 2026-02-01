---
title: '`pytch` v2: A Real-Time Monitoring Tool For Polyphonic Singing Performances'
tags:
  - Python
  - Audio Processing
  - Music Information Retrieval
  - Singing Voice Analysis
  - GUI
authors:
  - name: Sebastian Rosenzweig
    orcid: 0000-0003-4964-9217
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Marius Kriegerowski
    orcid:
    equal-contrib: true
    corresponding: false
    affiliation: 2
  - name: Frank Scherbaum
    orcid: 0000-0002-5050-7331
    corresponding: false
    affiliation: 3


affiliations:
 - name: Independent Researcher, Barcelona, Spain
   index: 1
 - name: Independent Researcher, Berlin, Germany
   index: 2
 - name: University of Potsdam, Potsdam, Germany
   index: 3
date: 1 February 2026
bibliography: paper.bib
---

# Summary
Polyphonic singing is one of the most widespread forms of music-making. During a performance, singers must constantly adjust their pitch to stay in tune with one another — a complex skill that requires extensive practice. Research has shown that pitch monitoring tools can assist singers in fine-tuning their intonation during a performance [@BerglinPD22_VisualFeedback_JPM]. Specifically, real-time visualizations of the fundamental frequency (F0), which represents the pitch of the singing voice, help singers assess their pitch relative to a fixed reference or other voices.
To support the monitoring of polyphonic singing performances, we developed `pytch`, an interactive Python tool with a graphical user interface (GUI) designed to record, process, and visualize multiple voices in real time. The GUI displays vocal spectra and estimated F0 trajectories for all singers, as well as the harmonic intervals between them. Additionally, users can adjust visual and algorithmic parameters interactively to accommodate different input devices, microphone signals, singing styles, and use cases.
Our tool builds upon a late-breaking demo in [@KriegerowskiS_Pytch_2017], which we refer to as version 1. Since its initial release, the tool has been significantly extended with a new real-time graphics engine, a modular audio processing backend that facilitates the integration of additional algorithms, and improved support for a wider range of platforms and recording hardware, which we refer to as version 2. The applications of `pytch` range from research in the field of computational musicology to pedagogical contexts focused on intonation and harmonic listening.


# Statement of Need
Software that assesses the pitch of a singing voice in real time is best known from Karaoke singing applications, such as Let's Sing[^1], Rock Band[^2], or Cantamus[^3]. These tools typically compare the singer’s pitch to a score reference to judge whether notes are ‘correct’ or ‘incorrect’. However, such applications face several limitations when applied to polyphonic or group singing contexts. Most notably, many Karaoke systems can only process one or two singing voices at a time, which is problematic for monitoring group performances. Additionally, software that relies on a score as a reference poses challenges for a cappella performances, where singers may drift together in pitch over time while maintaining relative harmony, or in orally-transmitted traditions that may lack a formal score altogether. Finally, existing open-source research software for singing voice processing, like Praat [@Boersma01_Praat_GI], Sonic Visualiser [@CannamLS10_SonicVisualizer_ICMC], and Tarsos [@SixCL13_Tarsos_JNMR], lack real-time feedback, preventing an effective feedback loop between singers and their tool.

To address these challenges, we developed `pytch`. Our tool is currently the only software that enables singers and conductors to monitor and train harmonic interval singing in real time — a skill that is essential in many vocal traditions. This includes not only polyphonic genres such as traditional Georgian vocal music [@ScherbaumMRM19_MultimediaRecordings_FMA] or Barbershop singing [@HagermanS80_Barbershop_CITESEER], where precise tuning between voices is stylistically central, but also the practice of non-tempered tuning systems found in various oral traditions. Unlike many existing tools, `pytch` does not require a musical score. The vocal spectra can help singers fine-tune the expression of formant frequencies, while melodic and harmonic issues become visible through F0 trajectories and harmonic intervals.

[^1]: <https://en.wikipedia.org/wiki/Let%27s_Sing>
[^2]: <https://en.wikipedia.org/wiki/Rock_Band_4>
[^3]: <https://cantamus.app>


# Research Impact Statement
Over its seven years of development to now, `pytch` has been tested and iterated through use in ensemble rehearsals, singing workshops, and ethnomusicological field research. For example, the tool was used to analyze the Sardinian singing style of the "Quintina" in which four singers manage to fuse the high frequency partials of their voices in such a way that an apparent fifth voice appears (see demo video[^4]). Furthermore, `pytch` has been used during a field expedition to Georgia, where ethnomusicologists recorded and analyzed traditional vocal music which belongs to the UNESCO intangible world cultural heritage (see demo video[^5]).

In addition to its use in musicological research, `pytch` also provides a platform for music information retrieval (MIR) research on real-time audio processing. Working with real-time data introduces challenges such as a limited audio context for analysis and strict timing constraints to ensure low-latency processing. Researchers can use `pytch` to develop, test, and compare algorithms for F0 estimation and other music information retrieval tasks [@StefaniT22_RealTimeMIR_DAFX;@Goto04_RealTimeF0_SC;@MeierCM24_RealTimePLP_TISMIR]. A notable example is the `rtswipe` [@MeierSSMB25_RealTimeSWIPE_CMMR] library, which was inspired by `pytch` use cases and integrated after its initial release. Using its implementation significantly increased the speed and accuracy of F0 estimates in `pytch`; consequently, it was selected as the default algorithm in our tool from version 2.3 onward.

[^4]: <https://www.uni-potsdam.de/de/soundscapelab/computational-ethnomusicology/the-benefit-of-body-vibration-recordings/real-time-analysis-of-larynx-microphone-recordings>
[^5]: <https://youtu.be/LPt83Wqf2e4>


# Multitrack Singing Recordings

To fully leverage the capabilities of `pytch`, it is essential to record each singer with an individual microphone. While there is no hard limit on the number of input channels, we recommend recording up to four individual singers to ensure visibility of the charts and responsiveness of the GUI. Stereo recordings—-such as those captured by a room microphone placed in front of the ensemble--are not suitable for the analysis with `pytch`, because contributions of individual voices are difficult to identify from polyphonic mixtures [@Cuesta22_Multipitch_PhD]. Suitable multitrack recordings can be obtained using handheld dynamic microphones or headset microphones. However, these setups are prone to cross-talk, especially when singers are positioned close together.

One way to reduce cross-talk is to increase the physical distance between singers or to record them in isolation. However, this is not always feasible, as singers need to hear one another to maintain accurate tuning. An effective workaround is the use of contact microphones, such as throat microphones, which capture vocal fold vibrations directly from the skin of the throat. This method offers a significant advantage: the recorded signals are largely immune to interference from other singers, resulting in much cleaner, more isolated recordings. Throat microphones have successfully been used to record vocal ensembles in several past studies [@Scherbaum16_LarynxMicrophones_IWFMA].

In addition to live monitoring, `pytch` can also be used to analyze pre-recorded multitrack singing performances. By playing back individual vocal tracks in a digital audio workstation (DAW) and using virtual audio routing tools such as Loopback[^6] (macOS) or BlackHole[^7], these tracks can be streamed into `pytch` as if they were live microphone inputs. This setup, which was also used in the demo video[^5], allows users to benefit from `pytch`’s real-time visualization and analysis features during evaluation of rehearsals, performances, or field recordings.

[^6]: <https://rogueamoeba.com/loopback/>
[^7]: <https://existential.audio/blackhole/>

# Software Design
`pytch` is implemented in Python, the predominant language for research software in music computing, which facilitates contributions from other researchers. The `pytch` codebase systematically separates GUI code from audio processing code to ease maintenance.

## Audio Processing
The real-time audio processing pipeline implemented in the file `audio.py` is the heart of `pytch` and consists of two main stages: recording and analysis. The recording stage captures multichannel audio waveforms from the soundcard or an external audio interface using the `sounddevice` library[^8]. The library is based on PortAudio[^9] and supports a wide range of operating systems, audio devices, and sampling rates. The recorded audio is received in chunks via a recording callback and fed into a ring buffer shared with the analysis process. When the buffer is sufficiently filled with audio chunks, the analysis process reads the recorded audio to compute several audio features.

For each channel, the analysis stage computes the audio level in dBFS, a time--frequency representation of the audio signal via the Short-Time Fourier Transform (see [@Mueller21_FMP_SPRINGER] for fundamentals of music processing), and an estimate of the F0 along with a confidence value. There are currently two F0 algorithms available: YIN [@CheveigneK02_YIN_JASA] from the `libf0` library [@RosenzweigSM22_libf0_ISMIR-LBD], and a real-time version of SWIPE [@CamachoH08_SawtoothWaveform_JASA] from the `rtswipe` library [@MeierSSMB25_RealTimeSWIPE_CMMR]. YIN is a time-domain algorithm that computes the F0 based on a tweaked auto-correlation function. It is computationally efficient and well-suited for low-latency applications, but it tends to suffer from estimation errors, particularly confusions with higher harmonics such as the octave. SWIPE is a frequency-domain algorithm that computes the F0 by correlating pitch candidate kernels derived from a sawtooth waveform with the spectrum of an input signal. It is known to be more robust to noise as YIN.

The obtained F0 estimates, which are natively computed in the unit Hz, are converted to the unit cents using a user-specified reference frequency. Depending on the audio quality and vocal characteristics, F0 estimates may exhibit artifacts such as discontinuities or pitch slides, which can make the resulting trajectories difficult to interpret [@RosenzweigSM19_StableF0_ISMIR]. Previous research has shown that using throat microphones can improve the isolation of individual voices in group singing contexts, resulting in cleaner signals and more accurate F0 estimates [@Scherbaum16_LarynxMicrophones_IWFMA]. To further enhance interpretability, `pytch` includes several optional post-processing steps: a confidence threshold to discard estimates with low confidence score, a median filter to smooth the trajectories, and a gradient filter to suppress abrupt pitch slides. As a final step in the audio analysis, the harmonic intervals between the F0 trajectories are computed. Every audio feature is stored separately in a dedicated ring buffer. After processing, the pipeline sets a flag that notifies the GUI that new data is ready for visualization.

## Graphical User Interface (GUI)
In this section, we provide a step-by-step explanation of the `pytch` GUI implemented in the file `gui.py`. Inspired by other scientific Python-based tools [@Aubier25_FUS_JOSS;@PatakyND19_mwarp1d_JOSS], we designed the `pytch` GUI using the open-source library `pyqtgraph`[^10].

[^8]: <https://github.com/spatialaudio/python-sounddevice>
[^9]: <https://www.portaudio.com>
[^10]: <https://www.pyqtgraph.org>

Right after the program start, a startup menu opens in which the user is asked to specify the soundcard, input channels, sampling rate, and window size for processing (see Figure \autoref{fig:menu}). Furthermore, the user can choose to store the recorded audio and the F0 trajectories on disk.

![`pytch` startup menu.\label{fig:menu}](../pictures/menu.png){ width=50% }

These configuration choices are required to initialize the audio processing module and the main GUI, which is loaded when the user clicks "ok". A screenshot of the main GUI which opens after successful initialization is shown in Figure \autoref{fig:GUI}.

![`pytch` GUI monitoring three singing voices.\label{fig:GUI}](../pictures/screenshot.png){ width=100% }

The main GUI is organized into three horizontal sections. On the left, a control panel provides a start/stop button and allows users to adjust both the visual layout and algorithmic parameters. The central section displays "channel views"--one for each input channel--color-coded for clarity. Each view includes a microphone level meter, a real-time spectrum display with a vertical line marking the current F0 estimate, and a scrolling spectrogram with a 5 second time context. Channels are listed from top to bottom in the order they were selected during setup. Optionally, the bottommost view can display a product signal from all channels.

The right section, referred to as the "trajectory view," provides time-based visualizations of either the F0 trajectories ("pitches" tab) or the harmonic intervals between voices ("differential" tab) with a 10 second time context. Using the controls in the left-side menu, the user can select the F0 estimation algorithm and improve the real-time visualization by adjusting the confidence threshold, the median filter length for smoothing, and the tolerance of the gradient filter. F0 and interval trajectories can be displayed with respect to a fixed reference frequency or a dynamic one derived from a selected channel, the lowest, or highest detected voice. Axis limits for this section can also be manually set.

# AI Usage Disclosure
The authors used the AI tool ChatGPT to assist with grammar, spelling, and language refinement in this article. In addition, ChatGPT was used to generate example code for PyQt6 functions and classes. No code was copied verbatim from ChatGPT; all generated examples were reviewed, adapted, and integrated by the authors.

# Acknowledgements
We would like to thank Lukas Dietz for his help with the implementation, Peter Meier, Sebastian Strahl, Simon Schwär, and Meinard Müller for the collaboration on the real-time SWIPE implementation, and all the singers who contributed to testing `pytch` during its development.

# References
