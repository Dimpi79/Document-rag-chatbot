# Mini-RAG Evaluation (5 Queries)

Document indexed: Research_Paper.pdf  
RAG settings: chunk=900 chars, overlap=120 chars, retriever top-k=8, rerank top-n=4  
Expected behavior: grounded answers with inline citations like [1], [2]. If not found, respond "Not found in the provided documents."

---

## Q1
**Question:** What is the main goal of the proposed system in this paper?  
**Expected:** Use a rate 1/3 convolutional encoder at the transmitter and an ANN-based autoencoder at the receiver to recover the original transmitted signal under noisy channel conditions, improving reconstruction performance.  
**Result:** Pass 
          The overall objective of the proposed system is to analyze how effectively an Artificial Neural Network (ANN) autoencoder can recover the transmitted signal under noisy channel conditions [1]. This system was implemented to study the ANN-based autoencoder's effectiveness for error correction in digital communication systems [3]. The main training goal of the ANN autoencoder is specifically to reduce the reconstruction error of the transmitted signal [2]. 
**Notes:**

---

## Q2
**Question:** What convolutional code rate and constraint length are used, and how many output bits are produced per input bit?  
**Expected:** Rate = 1/3; constraint length K = 5 (four memory registers); for each input bit, three output bits (X1, X2, X3) are generated.  
**Result:** Pass 
            The convolutional code rate used is 1/3. The encoder operates with a constraint length of K=5 [1].For every single input bit, three output bits (X1,X2,X3) are generated [1].
**Notes:**

---

## Q3
**Question:** What channel and modulation are used in the system model?  
**Expected:** BPSK modulation is used, and the signal is transmitted through an AWGN channel.  
**Result:** Pass  
           The bitstream is modulated using Binary Phase Shift Keying (BPSK) [2]. This process converts the digital data into a suitable form for transmission over the channel [2]. The modulated signal is then transmitted through an Additive White Gaussian Noise (AWGN) channel [2]. The system model represents a complete digital communication system that includes this noisy transmission channel [1]. 
**Notes:**

---

## Q4
**Question:** What is the reported MSE improvement after ANN-based reconstruction, and what are the MSE values before and after?  
**Expected:** Noisy signal MSE ≈ 0.245; ANN-decoded MSE ≈ 0.135; ~44.9% reduction in MSE.  
**Result:** Pass 
            The ANN-based reconstruction achieved a 44.9% reduction in Mean Square Error (MSE) [2][3]. The MSE value for the noisy input signal (before reconstruction) was approximately 0.245 [2]. After applying the ANN model, the MSE for the ANN-decoded signal (after reconstruction) was approximately 0.135 [2]. This reduction indicates that the reconstructed signal is closer to the original clean signal than the noisy input [1].  
**Notes:**

---

## Q5
**Question:** Why did BER not improve even though MSE improved, and what future work is suggested?  
**Expected:** ANN trained to minimize MSE focuses on waveform reconstruction; reconstructed signal can cross the bit decision threshold incorrectly, increasing BER. Suggested future work: bit-level training / treat decoding as classification (e.g., binary cross-entropy, classification-based approaches) to improve BER.  
**Result:** Pass 
             The Bit Error Rate (BER) did not show similar improvement even though the Mean Square Error (MSE) was reduced by approximately 44.9% [1]. This mismatch arose because minimizing MSE alone emphasizes reconstruction accuracy rather than direct optimization of symbol-level or bit-level correctness [2]. In several cases, the reconstructed signal crossed the bit decision threshold incorrectly, resulting in a higher BER and additional bit errors [1][3].

             This behavior suggests that training the network only to minimize MSE is insufficient for reliable digital decoding [1]. Future work should focus on bit-level training and classification-based approaches to improve decoding accuracy [1].
**Notes:**
