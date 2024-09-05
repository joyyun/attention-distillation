# CS224U-Project

# Reading Between the LIMEs: Understanding the Transfer of Language Understanding Through Logit- and Attention-Based Distillation
### Authors: Joy Yun, Hanna Lee, Zhiyin Lin
(All authors contributed equally.)

CS 224U (Spring 2023) final project.

[Paper](CS224u_final_paper.pdf)

## Abstract
Previous work has shown that knowledge distillation is an effective technique for improving small-model natural language benchmark performance by training it to emulate a large teacher model's predictions. There is little clarity around whether a teacher model's deep understanding of language is truly distilled and represented within student models or if the student models are simply learning heuristics to match the teacher's outputs. In this paper, we explore whether using attention-based knowledge distillation, in place of "vanilla" logit-based distillation, can guarantee a more thorough transfer of linguistic logic and understanding from teacher to student. We provide evaluations of alignments using quatitative metric Cohen's Kappa and qualitative LIME analysis. We find that 1) models can attain the same performance accuracy on the CoLA benchmark with very misaligned reasoning, 2) 
performing attention-based distillation using different attention layers can lead to significantly different logical alignments between the student and teacher model, and 3) attention-based distillation using first-layer attention most effectively transfers feature-level reasoning from the teacher to the student.
