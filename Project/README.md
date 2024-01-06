# DeepFake detection as an anomaly detection problem

## Project description

Due to advances in computer vision and most importantly deep learning considerable advances have been made in the area of photo realistic face manipulation. So-called DeepFakes represent manipulated videos of faces where the original appearance of a person is replaced with the appearance of some selected target face. These DeepFakes represent a considerable societal problem, as they can be used to produce fake news, spread misinformation and adversely affect the trust of people in news media coverage and reporting.

As a result, considerable research effort is being directed towards designing DeepFake detectors that can reliably detect wether a video is fake or real. Several such detectors have been porposed in the literature, but these still exhibit difficulties in generalization performance. In most cases, the detectors work well for detecting a specific form of DeepFake, generated with some specific algorithm, but fail to generalize to new, unseen DeepFake types. This is best shown by cross-dataset experiments, or cross-manipulation type of experiments, where a trained detector works well on one dataset/method, but not on another.

The goal of this project is to evaluate a DeepFake detector around deep learning models trained for anomaly detection and compare it to a discriminatively trained detector. Anomaly detection models are trained only on examples of one class (e.g., real face in this problem domain) and then try to to flag all data that does not conform with the learned class as anomalous (e.g., as a fake in this case). By using one-class classification models the idea is to produce DeepFake detectors that generalize better across datasets and DeepFake types.

Tasks:
- Identify one 1-class DeepFake detection model (SeeABLE, OC-FD) and 1 discriminatively trained DeepFake detector (SBI, Two-stream, etc.) and set it up
- Identify a suitable deep fake dataset for your experiments - FaceForensics++
- Set up the anomaly detection framework
- Set up the baseline DeepFake detection technique to compare with
- Test the performance and evaluate generalization capability of your model on cross-manipulation experiments and analyze data characteristics (DeepFake quality, gaze, type, etc.), run-time requirements, performance, ...

## Project dependencies

TODO: Write the rest of the report in Latex and Markdown, work in progress

Detectors currently used are the following:

- OC-FakeDect (Trained on FF++)
- SeeABLE (Trained on FF++)
- SelfBlendedImages (Trained on FF++)
- Two-stream (Trained on FF++)

All datasets current access requirements:

Maybe access:

- UADFV: idk
- EBV: No link found 404
- DFFD:  Request access
- WildDeepFake: Request access
- Celeb-DF: Request access
- DeeperForensics-1.0: Request access
- FFIW: Request access

Accessible:

- ForgeryNet: Free access
- FFHQ, iFakeFaceDB: Free access
- DFDC: Too big 470GB (Able to get preprocessed images from friend, halfway acquired)
- DF-TIMIT: Already acquired
- FaceForensics++: Already acquired


## References

[1] H. Dang, F. Liu, J. Stehouwer, X. Liu, and A. K. Jain. On the detection of digital face manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern recognition, pages 5781–5790, 2020.

[2] B. Dolhansky, J. Bitton, B. Pflaum, J. Lu, R. Howes, M. Wang, and C. C. Ferrer. The deepfake detection challenge (dfdc) dataset. arXiv preprint arXiv:2006.07397, 2020.

[3] Y. He, B. Gan, S. Chen, Y. Zhou, G. Yin, L. Song, L. Sheng, J. Shao, and Z. Liu. Forgerynet: A versatile benchmark for comprehensive forgery analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4360–4369, 2021.

[4] L. Jiang, R. Li, W. Wu, C. Qian, and C. C. Loy. Deeperforensics-1.0: A large-scale dataset for real-world face forgery detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2889–2898, 2020.

[5] T. Karras, S. Laine, and T. Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019.

[6] H. Khalid and S. S. Woo. Oc-fakedect: Classifying deepfakes using one-class variational autoencoder. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 656–657, 2020.

[7] P. Korshunov and S. Marcel. Deepfakes: a new threat to face recognition? assessment and detection. arXiv preprint arXiv:1812.08685, 2018.

[8] N. Larue, N.-S. Vu, V. Struc, P. Peer, and V. Christophides. Seeable: Soft discrepancies and bounded contrastive learning for exposing deepfakes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21011–21021, 2023.

[9] Y. Li, M.-C. Chang, and S. Lyu. In ictu oculi: Exposing ai generated fake face videos by detecting eye blinking. arXiv preprint arXiv:1806.02877, 2018.

[10] Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu. Celeb-df: A large-scale challenging dataset for deepfake forensics. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3207–3216, 2020.

[11] A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner. Faceforensics++: Learning to detect manipulated facial images. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1–11, 2019.

[12] K. Shiohara and T. Yamasaki. Detecting deepfakes with self-blended images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18720–18729, 2022.

[13] C. Shuai, J. Zhong, S. Wu, F. Lin, Z. Wang, Z. Ba, Z. Liu, L. Cavallaro, and K. Ren. Locate and verify: A two-stream network for improved deepfake detection. In Proceedings of the 31st ACM International Conference on Multimedia, pages 7131–7142, 2023.

[14] X. Yang, Y. Li, and S. Lyu. Exposing deep fakes using inconsistent head poses. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8261–8265. IEEE, 2019.

[15] T. Zhou, W. Wang, Z. Liang, and J. Shen. Face forensics in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5778–5788, 2021.

[16] B. Zi, M. Chang, J. Chen, X. Ma, and Y.-G. Jiang. Wilddeepfake: A challenging real-world dataset for deepfake detection. In Proceedings of the 28th ACM international conference on multimedia, pages 2382–2390, 2020.

[17] D. A. Coccomini, G. K. Zilos, G. Amato, R. Caldelli, F. Falchi, S. Papadopoulos, and C. Gennaro. Mintime: Multi-identity size-invariant video deepfake detection, 2022.