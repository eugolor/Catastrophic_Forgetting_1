# Catastrophic_Forgetting_1

Three notebooks:

visual_CF established that catastrophic forgetting is real, dramatic, and measurable, a 99% to 0% cliff in a controlled setting.

llm_FT_forgetting showed the same phenomenon operating subtly inside a real language model, connecting the toy example to production concerns.

Adapter based CL demonstrated the technique production teams actually use, adapter-based fine-tuning, and quantified why it works: 520× storage reduction, zero general knowledge degradation, and per-task modularity.



# Part 1: Seeing It Happen
The first step was making catastrophic forgetting visible in the simplest possible setting.
A small neural network was trained on MNIST digits 0–4 (Task A), reaching 99% accuracy over 5 epochs. Training was then switched to digits 5–9 (Task B), while continuing to measure accuracy on Task A.
Results:

The drop was total and immediate. Task A accuracy went from 99% to exactly 0% because the weights that encoded "what a 3 looks like" were completely overwritten to encode "what a 7 looks like."
The 0% result (rather than ~20% random chance) is meaningful: the model wasn't guessing randomly. it was confidently predicting digits 5–9 for every Task A image. That's worse than chance.
Fix attempted — Elastic Weight Consolidation (EWC): After Task A training, the Fisher information matrix was used to identify which weights were most critical for Task A. A penalty term was added to the Task B loss function to resist moving those weights. The result: Task A accuracy was substantially preserved during Task B training, demonstrating the core principle — protect what matters, adapt the rest.

# Part 2: From Toy Example to Real Language Models
The MNIST result was dramatic, but it raised a question: does this happen in real LLMs too, or just in small demo networks?
To find out, DistilGPT2 (82M parameters) was fine-tuned on 500 medical Q&A examples from the MedAlpaca dataset over 3 epochs. Before and after fine-tuning, perplexity was measured on a fixed set of general knowledge prompts, sentences like "The capital of France is" and "Python is a programming language."
Perplexity measures how confused the model is about a piece of text. Lower is better. A rise in perplexity after fine-tuning means the model finds previously familiar text less natural, a signal of forgetting.

Selected results:
| Prompt | Before | After | Change |
|---|---|---|---|
| "The capital of France is..." | 5.71 | 5.92 | +0.21 |
| "Albert Einstein was born in..." | 5.05 | 5.52 | +0.47 |
| "Python is a programming language..." | 3.28 | 3.61 | +0.33 |
| "The speed of light is approximately..." | 5.05 | 4.49 | −0.56 |

The degradation was real but uneven. Most general knowledge topics worsened slightly. Interestingly, one prompt — the speed of light — actually improved after medical fine-tuning, likely because medical text is dense with precise numerical facts, which reinforced that sentence pattern. This is called positive transfer.
The key finding: even with only 500 examples and 3 epochs, fine-tuning on a narrow domain measurably disrupts general knowledge. At production scale, tens of thousands of examples, many more epochs, this degradation would be far more severe.


# Part 3: How the Industry Solves It
The root cause of catastrophic forgetting is that full fine-tuning modifies all model weights. The industry solution is to not modify the base model at all.
LoRA (Low-Rank Adaptation) keeps the base model completely frozen and instead adds tiny trainable "adapter" matrices alongside each layer. Instead of updating an 82M-parameter model, you train roughly 150K adapter parameters, about 0.18% of the model.
Each task gets its own adapter. Switching tasks means swapping a small file rather than reloading a full model copy.

Results on the same DistilGPT2 base:
| Metric | Full Fine-tuning | LoRA Adapter |
|---|---|---|
| Parameters trained | 81,912,576 | 147,456 |
| Storage per task | 312.5 MB | 0.6 MB |
| Storage ratio | 1× | 520× smaller |
| General knowledge perplexity change | increases | near zero |
| Medical perplexity change | improves | improves similarly |

The perplexity chart told the clearest story: after LoRA fine-tuning, general knowledge perplexity did not rise. It stayed flat or marginally improved, identical to baseline. Zero forgetting, because the base model was never touched.
At 10 tasks, full fine-tuning requires 3.1GB of model storage. LoRA requires 6MB of adapters plus one shared 312MB base, roughly 9× less total, with the gap widening with every additional task.
