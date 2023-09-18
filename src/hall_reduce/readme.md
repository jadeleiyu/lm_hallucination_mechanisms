```cap```: Our **C**heckpoint **A**ttribution **P**atching (**CAP**) method of reducing LM factual hallucinations.

```baselines```: Several baseline models to compare on ZsRE/NaturalQA/TriviaQA datasets:

  1. Knowledge Neurons (KN)
  2. Knowledge Editor (KE)
  3. Model Editing via Gradient Decomposition (MEND)
  4. Rank-One model editing (ROME)  
  
  **Note:** ROME cannot be evaluated on NaturalQA/TriviaQA because it requires explicit subject tokens in queries, which neither datasets have.
  **Note:** Another baseline method, Inference Time Intervention (ITI) can be implemented as a simplified version of our CAP model.
 

