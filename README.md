


# Contributions

In this project, I significantly expanded upon Valerio Valardo's Sound of AI by integrating emotional context into melody generation. Specifically, my contributions include:

**Emotion Integration**: Incorporated emotion-based conditioning using Russell's 4Q emotional model (Q1: happy, Q2: angry, Q3: sad, Q4: relaxed) to intentionalize the melodic output's impact

**Dataset Enhancement**: Originally 6-Melody Json File replaced by the EMOPIA .midi file dataset (1000+), categorized by emotion quadrants, to train the model effectively in emotion-specific music generation.

**Seed Music Implementation**: Implemented functionality to accept user-provided seed MIDI files, enabling the model to generate emotionally coherent continuations or variations based on the emotional quadrant selected by the user.

**Model Training**: Retrained and fine-tuned the Transformer-based model from Sound of AI using the emotion-labeled EMOPIA dataset alongside seed MIDI inputs to achieve nuanced, emotion-driven melody generation.

Through these contributions, the updated model demonstrates improved emotional expressivity and adaptability in generated melodies.


<img width="533" alt="Screenshot 2025-03-13 at 11 23 22 AM" src="https://github.com/user-attachments/assets/bd62aaa7-0e17-40e6-890b-38b6b2ec1968" />


## Acknowledgements

### Dataset - EMOPIA Dataset
Attribution: H.T.Hung and J. Ching, and S.H Doh, “EMOPIA: A Multimodal Pop Piano Dataset for Emotion Recognition and Emotion-based
Music Generation”, in Proc. of the 22nd Int. Society for Music Information Retrieval Conf., Online, 2021.

MIT License - Copyright (c) 2021 Hsiao-Tzu Hung

### Code Adapted and Modified from - Valerio Velardo's repository](https://github.com/musikalkemist/generativemusicaicourse/tree/main?tab=readme-ov-file#readme-ov-file), Original Author: Valerio Velardo ([GitHub](https://github.com/musikalkemist))
MIT License - Copyright (c) 2023 Valerio Velardo



MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
