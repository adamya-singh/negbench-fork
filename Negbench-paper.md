Vision-Language Models Do
Not
Understand Negation
Kumail Alhamoud
1
Shaden Alshammari
1
Yonglong Tian
2
Guohao Li
3
Philip H.S. Torr
3
Yoon Kim
1
Marzyeh Ghassemi
1
1
MIT
2
Google DeepMind
3
University of Oxford
https://NegBench.github.io
Abstract
Many practical vision-language applications require mod-
els that understand
negation
, e.g., when using natural lan-
guage to retrieve images which contain certain objects but
not others. Despite advancements in vision-language mod-
els (VLMs) through large-scale training, their ability to
comprehend negation remains underexplored. This study
addresses the question: how well do current VLMs under-
stand negation? We introduce NegBench, a new bench-
mark designed to evaluate negation understanding across
18 task variations and
79
k examples spanning image, video,
and medical datasets. The benchmark consists of two core
tasks designed to evaluate negation understanding in di-
verse multimodal settings: Retrieval with Negation and
Multiple Choice Questions with Negated Captions. Our
evaluation reveals that modern VLMs struggle signicantly
with negation, often performing at chance level. To address
these shortcomings, we explore a data-centric approach
wherein we netune CLIP models on large-scale synthetic
datasets containing millions of negated captions. We show
that this approach can result in a 10% increase in recall on
negated queries and a 40% boost in accuracy on multiple-
choice questions with negated captions.
1. Introduction
Joint embedding-based Vision-Language Models (VLMs),
such as CLIP, have revolutionized how we approach multi-
modal tasks by learning a shared embedding space where
both images and text are mapped together. This shared
space enables a variety of applications, including cross-
modal retrieval, video retrieval, text-to-image generation,
image captioning, and even medical diagnosis [
2
,
 18
,
 19
,
21
,
 30
,
 32
,
 35
,
 38
Œ
40
,
 49
]. By aligning visual and linguis-
tic representations, these models achieve remarkable per-
formance across domains and are able to model complex
interactions between vision and language inputs.
Despite these advances, there is an emerging limita-
Figure 1. We present
NegBench
with image retrieval and multiple-
choice tasks to evaluate negation understanding. CLIP-based mod-
els frequently misinterpret negation in both tasks, but we show
how a synthetic data approach can improve performance.
tion: these models fail to handle
negation
, which is es-
sential in many real-world scenarios. Negation enables
precise communication by specifying what is false or ab-
sent [
12
,
 16
,
 26
,
 27
]. For example, a radiologist may search
for images showing ﬁbilateral consolidation with no evi-
dence of pneumoniaﬂ, or a safety inspector might query
ﬁconstruction sites with no barriersﬂ. Current benchmarks
like CREPE and CC-Neg have introduced limited tests of
negation, but they rely on rigid, templated examples that
do not reect the complexity of natural language queries
[
24
,
 41
]. As a result, they fall short in evaluating how well
VLMs understand negation in practical applications.
To comprehensively evaluate how well VLMs handle
negation, we design a multi-level evaluation paradigm in-
spired by real-world information retrieval systems, where a
coarse-grained retrieval step often precedes a ne-grained
ranking or selection step [
23
,
 29
].
The rst task, Retrieval-Neg, tests whether models can
handle real-world queries that mix afrmative and negative
statements, such as ﬁa beach with no peopleﬂ or ﬁa build-
ing without windows.ﬂ This task challenges the model to
retrieve images from diverse datasets based on the presence
1
arXiv:2501.09425v1  [cs.CV]  16 Jan 2025


of certain elements and the absence of others, simulating
scenarios found in search engines, content moderation, and
recommendation systems. By retrieving several potentially
relevant matches (e.g., top-5 retrieval), Retrieval-Neg serves
as the coarse-grained retrieval component of our evaluation.
The second task, MCQ-Neg, provides a ne-grained,
structured evaluation that directly assesses specic failures
in negation. In this task, the model must choose the cor-
rect description of an image from several closely related
options, where the incorrect choices are hard negatives, dif-
fering only by what is afrmed or negated. For instance, in
medical diagnostics, consider distinguishing between ﬁThe
X-ray shows evidence of pneumonia but no evidence of
pleural effusionﬂ and ﬁThe X-ray shows evidence of pleural
effusion but no evidence of pneumonia.ﬂ These statements
are linguistically similar but convey opposite diagnoses, re-
quiring the model to parse subtle yet critical differences.
Through our evaluation pipeline, we uncover a surprising
limitation: joint embedding-based VLMs frequently col-
lapse afrmative and negated statements into similar em-
beddings, treating ﬁa dogﬂ and ﬁno dogﬂ as nearly indis-
tinguishable. This afrmation bias reveals a signicant
shortcoming that was not sufciently addressed in previous
benchmarks like CREPE or CC-Neg.
Recognizing this critical gap, we then ask: If current
models fail to understand negation, can we improve them?
To tackle this, we propose a data-centric solution, introduc-
ing two large-scale synthetic datasetsŠCC12M-NegCap
and CC12M-NegMCQŠdesigned to improve negation
comprehension. Fine-tuning CLIP-based models on these
datasets leads to substantial improvements, including a 10%
increase in recall on negated queries and a 40% boost in ac-
curacy on multiple-choice questions with negated captions.
The rest of the paper follows a challenge-diagnosis-
solution structure. We introduce NegBench to evaluate
negation comprehension, analyze VLMs' afrmation bias,
and propose a data-driven solution using synthetic negation
examples. We will open-source all models and data to foster
research in negation understanding and its applications.
2. Related Work
Our work lies within the eld of evaluating and advanc-
ing foundational vision-language models (VLMs). Joint-
embedding models based on CLIP [
31
] show impressive
generalization across visio-linguistic tasks like cross-modal
retrieval, image captioning, and visual question answering
[
2
,
 18
,
 19
,
 30
,
 32
 ,
 35
,
 38
Œ
40
] in diverse visual domains,
extending beyond natural images to videos and medical im-
ages [
3
,
 13
,
 21
,
 22
,
 28
,
 49
]. We introduce a benchmark and
data-centric approach to rigorously evaluate and improve
negation understanding in these VLMs.
Negation Understanding in Language and Vision.
Re-
cent work showed that large language models perform sub-
optimally wh en tasked with negation understanding [
9
,
 45
].
We go a step further by showing that vision-language mod-
els exhibit a more severe afrmation bias, completely fail-
ing to differentiate afrmative from negative captions.
Despite this critical limitation, existing benchmarks pro-
vide limited assessments of negation in VLMs. CREPE [
24
]
and the concurrent work CC-Neg [
41
] are among the few
vision-language benchmarks that include negation, but they
focus on compositional understanding and rely on linguis-
tic templates that fail to reect the varied ways negation ap-
pears in real user queries. In contrast, our proposed bench-
mark, NegBench, leverages an LLM to generate natural-
sounding negated captions, spanning a broader range of
negation types and contexts across images, videos, and
medical datasets. This systematic design enables a thor-
ough evaluation of VLMs' ability to handle negation in mul-
timodal settings, uncovering unique challenges and failure
cases that have not been fully addressed in prior work.
Improving CLIP for Compositionality and Negation.
Recent methods have explored improving the generaliza-
tion abilities of CLIP-like VLMs for visio-linguistic com-
positionality and limited aspects of negation understand-
ing. For instance, NegCLIP [
48
] employs composition-
aware mining when netuning CLIP to enhance composi-
tional reasoning, while ConCLIP [
41
] modies the CLIP
loss to incorporate synthetic, template-based negation ex-
amples. In the medical domain, negation is a common fea-
ture in clinical text reports, often indicating the absence of
specic pathologies [
44
]. Specialized models like Biomed-
CLIP [
49
] and CONCH [
21
] have been pretrained on mil-
lions of biomedical image-text pairs to address a variety of
medical tasks, leveraging domain-specic knowledge from
large-scale multimodal data. NegBench provides a system-
atic way to evaluate general-purpose and medical VLMs.
Synthetic Data for Model Training.
It is common to use
synthetic data to improve the performance of models in
computer vision [
1
,
 5
,
 15
,
 47
]. Recent studies have shown
that it is possible to use synthetic data to learn general
vision-language representations, with some models trained
entirely on synthetic images and captions achieving results
comparable to real data [
11
,
 42
,
 43
]. Our approach is similar
in spirit, but it constructs synthetic datasets to teach models
a new, complex capabilityŠ
negation understanding.
3. The Negation Benchmark (NegBench)
We design NegBench as a multi-level evaluation to as-
sess the capacity of jo int-based vision-language models
to understand negation across different tasks: (1) coarse-
grained retrieval, by accurately retrieving images that sat-
isfy specied inclusions and exclusions, and (2) ne-
grained question-answering, by selecting the correct de-
scription from closely related options, testing the model's
detailed understanding of negation beyond simple retrieval.
2


Figure 2.
General Pipeline for Constructing NegBench
. We start by extracting positive concepts from vision datasets. An LLM proposes
negative concepts, which are veried with an object detector for datasets without explicit object annotations. We use templates to generate
captions with negation, then paraphrase them by an LLM to ensure linguistic variety and robust evaluation of negation understanding.
In the Retrieval-Neg task, the model retrieves the top-
5 images that match both afrmative and negative criteria
within a query. In the MCQ-Neg task, the model selects the
correct description of an image from options that differ only
in the afrmation or negation of specic elements.
3.1. Transforming Datasets for Negation Evaluation
General Dataset Transformation Overview.
To imple-
ment the two-stage evaluation pipeline of NegBench, we
adapt several popular vision datasets, covering images
(COCO [
20
], VOC2007 [
7
]), video (MSR-VTT [
46
]), and
specialized medical imaging domains (CheXpert [
14
]). For
each dataset, we identify positive elements
f
pos
g
, which
represent objects or concepts present in the image, and neg-
ative elements
f
neg
g
, which are absent from the image but
commonly associated with the present objects. When avail-
able, we use object-level annotations to identify these el-
ements, as in COCO, VOC2007, and CheXpert; for other
datasets, we derive positive and negative elements directly
from the captions. This exible approach allows NegBench
to extend any vision dataset, whether it includes object-level
annotations or captions, to evaluate negation comprehen-
sion across diverse tasks and data modalities.
In the Retrieval-Neg task, we modify standard cap-
tions by including negations, evaluating how models handle
queries that specify both present and absent elements. For
example, captions are modied as: ﬁThere is no
x
in the
image. [Original Caption].ﬂ or ﬁ[Original Caption]. There
is no
x
in the image.ﬂ To introduce linguistic diversity, we
use LLaMA 3.1 [
6
] to paraphrase these captions.
For the MCQ-Neg task, we generate multiple-choice
questions (MCQs) for each image. The model must identify
the correct description based on three linguistic templates:
Afrmation, Negation, and Hybrid [
17
].
1. Afrmation
: ﬁThis image includes
A
(and
C
).ﬂ
2. Negation
: ﬁThis image does not include
B
.ﬂ
3. Hybrid
: ﬁThis image includes
A
but not
B
.ﬂ
Each MCQ consists of one co rrect answer and three in-
correct answers, which serve as hard negatives, misleading
the model if it does not properly understand negation. A
correct answer accurately describes the presence of
f
pos
g
elements or negates
f
neg
g
elements. A False Afrma-
tion (e.g., ﬁThis image includes
x
ﬂ when
x
2 f
neg
g
) or
a False Negation (e.g., ﬂThis image does not include
x
ﬂ
when
x
2 f
pos
g
) highlights the model's failure to com-
prehend the image. The Hybrid template further evaluates
the model's ability to combine afrmation and negation in
the same caption. These MCQs are also paraphrased using
LLaMA 3.1 to increase linguistic diversity.
3.2. Applicability Across Data Types and Domains
NegBench supports a wide range of data types and domains,
enabling comprehensive negation evaluation.
Video Understanding.
Video retrieval tasks introduce tem-
poral complexity, where negation can involve both objects
and actions that vary over time. Using MSR-VTT as an ex-
ample, we prompt LLaMA 3.1 [
6
] to extract positive and
negative elements from each video's caption. These ele-
ments may represent either objects present in the video or
actions taking place. For Retrieval-Neg, we create cap-
tions specifying both the presence of some elements and
the absence of others (e.g., ﬁA person is cooking but not
eatingﬂ). In MCQ-Neg, we generate multiple-choice ques-
tions where the model must select the description that most
accurately represents a video segment, requiring it to reason
about negation of objects and actions in dynamic scenes.
Medical Image Interpretation with CheXpert.
Accurate
negation understanding is critical in high-stakes domains
like medical imaging. Using the CheXpert dataset [
14
], we
focus on the most frequent condition
Lung Opacity
and de-
sign two binary classication tasks:
Task 1: Afrmation Control Task.
This task evaluates the
model's ability to associate images with specic medical
conditions using afrmative statements.
3


Question
: Which option describes this image?
A) This image shows Lung Opacity.
B) This image shows Atelectasis.
Task 2: Negation Understanding Task.
This task tests
whether the model can correctly interpret negation, distin-
guishing the presence or absence of a medical condition.
Question
: Which option best describes the image?
A) This image shows Lung Opacity.
B) This image does
not
show Lung Opacity.
These extensions highlight the adaptability of NegBench
to various data types and domains, from general images and
videos to specialized medical imaging. This versatility en-
sures that NegBen ch provides rigorous, contextually rele-
vant evaluations of negation understanding in VLMs.
Figure 3.
Performance drop in recall@5 on (a) COCO and
(b) HardNeg-Syn text-to-image retrieval with negated captions
(green stars) compared to original captions (orange circles).
All models show substantial drops in performance, with NegCLIP
experiencing the largest drop of 23.0% on HardNeg-Syn, which
features hard negatives requiring stronger negation reasoning.
3.3. Synthetic Datasets for Controlled Evaluation
To rigorously test negation understanding, we construct
HardNeg-Syn
, a dataset that precisely controls object pres-
ence and absence by synthesizing hard negative images.
Motivation and Benets of Synthetic Data.
Syn-
thetic data offers several advantages over traditional im-
age datasets. First, by creating ﬁhard negativesﬂŠimage
pairs that differ only by a single object's presence or ab-
senceŠwe can evaluate the sensitivity of models to nega-
tion with minimal confounding variables. Additionally, im-
age datasets like COCO and VOC2007 are limited in the
range of visual concepts they cover; COCO has 80 objects
while VOC2007 includes only 20. To expand this diversity,
we prompt a large language model to propose a broader
set of objects, which we use as targets in our synthetic
dataset. This approach enables the generation of visually
varied scenes that more comprehensively test negation com-
prehension across a wider array of objects and contexts.
Construction Process for the HardNeg-Syn Evaluation
Dataset.
We create 10,000 image pairs using Stable Dif-
fusion [
34
], where each pair includes one image contain-
ing a target object and another where it is explicitly absent.
To ensure accurate object presence or absence, we use the
open-vocabulary object detector OWL-ViT [
25
].
4. NegBench Evaluations: Results and Insights
In this section, we benchmark the negation abilities of dif-
ferent VLMs using NegBench, comparing models based
on their architecture, training data, and training objectives
to reveal specic areas where negation understanding re-
mains limited. Specically, we evaluate ve CLIP ViT-B/32
models on Retrieval-Neg and MCQ-Neg tasks. These in-
clude OpenAI CLIP [
31
], CLIP-laion400m [
37
], and CLIP-
datacomp [
8
], which differ by pretraining dataset, as well
as NegCLIP [
48
], trained to improve compositional lan-
guage understanding, and ConCLIP [
41
], trained specif-
ically to improve negation understanding. To handle the
video dataset, MSR-VTT, we follow [
3
] and encode
4
uni-
formly sampled frames per video, averaging their features
to obtain the CLIP video embedding. For medical tasks, we
evaluate CONCH [
21
] and BioMedCLIP [
49
], two medical
foundation VLMs. We also assess the impact of scaling up
CLIP-laion400m (ViT-B, ViT-L, and ViT-H) to determine if
model size improves negation understanding.
CLIP models struggle with negated queries in retrieval
tasks.
We evaluate ve CLIP-based models on the origi-
nal COCO text-to-image retrieval task and its Retrieval-Neg
version, where captions include negated statements. Across
models, performance drops signicantly on the negated
task. In COCO retrieval (Figure
 3
a), CLIP-laion400m expe-
riences a 7.7% drop in recall@5, with CLIP-datacomp and
CLIP showing drops of 7.6% and 6.8%, respectively. In the
more challenging HardNeg-Syn retrieval task (Figure
 3
b),
the performance drops are even more pronounced due to the
presence of hard negatives,
i.e
. images that closely resemble
positive examples but differ by the exclusion of a single ob-
ject. Here, NegCLIP, despite its promise for compositional
understanding, suffers a 23.0% drop, while ConCLIP, de-
signed specically for negation understanding, still declines
by 18.0%. These results sugg est that interpreting negation,
particularly in the presence of hard negatives, remains a key
challenge for retrieval tasks.
MCQ-Neg reveals severe limitations in CLIP models.
4


Figure 4.
MCQ-Neg performance for (a) baseline CLIP models, (b) larger model sizes, and (c) medical VLMs.
(a) CLIP-based
models mostly perform worse than random guessing (shown as a red dashed line) on most datasets. (b) Scaling up CLIP models does not
signicantly improve negation understanding. (c) Medical VLMs experience a signicant drop in performance on negation MCQs.
Figure
 4
a shows that most models perform worse than ran-
dom guessing (indicated by the red dashed line at 25%) on
the MCQ-Neg task, with CLIP-base achieving only 15% on
COCO and 8% on VOC2007. These results reveal a fun-
damental limitation of CLIP's pretraining objective, which
encourages strong associations between visual concepts and
specic words, but struggles to interpret negation. Notably,
CLIP-laion400m performs better, reaching over 40% accu-
racy on the HardNeg-Syn dataset. This improvement likely
stems from the fact that both CLIP-laion400m and Stable
Diffusion (used to generate the HardNeg-Syn dataset) were
trained on the LAION dataset [
36
]. However, a score of
40% on a 4-way multiple-ch oice task is still far below an
acceptable level, demonstrating that even under this setup,
models exhibit a serious lack of negation understanding.
Scaling CLIP does not address the negation problem.
As shown in Figure
 4
b, scaling up the model size from
ViT-B/32 (86M parameters) to ViT-L/14 (307M parame-
ters) and ViT-H/14 (632M parameters) does not qualita-
tively improve negation understanding. While ViT-H/14
performs slightly better on COCO and VOC2007, it un-
derperforms on HardNeg-Syn and MSR-VTT compared to
ViT-B/32. These results suggest that increasing model size
alone is not an effective strategy for addressing the funda-
mental issues with negation understanding.
Critical failures in high-stakes medical tasks.
Figure
 4
c
presents the results for the CheXpert MCQ-Neg task, where
BioMedCLIP and CONCH exhibit substantial performance
drops of 24.6% and 33.2%, respectively, when negation is
introduced. This result is especially concerning in the con-
text of medical diagnostics, where accurate interpretation of
negation (e.g., the presence or absence of a condition such
as Lung Opacity) is essential for correct diagnoses and fa-
vorable patient outcomes.
4.1. Why Do VLMs
Not
Understand Negation?
The results from NegBench reveal that CLIP VLMs strug-
gle with different forms of negation understanding, moti-
vating a deeper analysis into the underlying causes of these
failures. In this section, we examine model performance
across different MCQ types and analyze the embedding
spaces of various models to uncover specic shortcut strate-
gies that limit their negation comprehension.
Model performance varies widely across MCQ types.
To understand why models perform below random chance,
we categorize the MCQs into three types based on the
correct answer template: Afrmation, Negation, and Hy-
brid. Figure
 5
 compares model accuracy across these MCQ
types, with evaluations conducted in two settings: one us-
ing LLaMA 3.1 to paraphrase answer choices into natural-
sounding sentences, and another using rigid linguistic tem-
plates. All models perform poorly on Negation MCQs, re-
ecting a general struggle with negation understanding.
Most models tend to select Negation sentences regard-
less of whether answers are templated or LLM-paraphrased,
as seen in the selection frequencies shown to the right of the
gure. This behavior likely arises from task design, where
67% of MCQs (Negation and Hybrid) lack a correct afr-
mative option, leading models to default to ﬁThis image
does not include
f
pos
g
.ﬂ These results suggest that mod-
els trained with CLIP-like objectives often adopt shortcut
strategies that ignore specic words like ﬁno.ﬂ
The template-based results reveal more biases in model
behavior. For instance, ConCLIP outperforms on Hybrid
MCQs, achieving the highest accuracy, but fails entirely
on Afrmation MCQs, scoring 0% on both image datasets.
This bias is particularly prominent in the rigid template
structure, where ConCLIP is skewed towards constructs like
ﬁThis image includes X but not Y.ﬂ In fact, as we will show
next, ConCLIP maps all templated Hy brid captions to the
same location in its embedding space.
Embedding analysis reveals VLM shortcut strategies.
To investigate potential shortcut strategies, we analyze the
embedding spaces of various models using 24 Afrmative
(ﬁXﬂ) and 24 Negated (ﬁNot Xﬂ) templates to create 48 cap-
tions per object. We apply PCA to the resulting embeddings
(Figure
 6a
). The templates are detailed in the appendix.
We observe varying behaviors across models. The over-
5


Figure 5.
Performance by MCQ type (Afrmation, Negation, Hybrid) across (a) template-based and (b) LLM-paraphrased answer
choices.
VLMs show signicant biases towards specic templates (
e.g
. ConCLIP with Hybrid). Template selection frequency (analyzed in
the appendix) conrms that CLIP defaults to Negation answers, especially when a positive object is incorrectly negated.
(a) PCA embeddings for afrmative (dots) and negated (triangles) captions.
(b) PCA embeddings for hybrid captions (diamonds) and cases where two objects are negated (stars) or afrmed (squares).
Figure 6.
PCA Projections of Caption Embeddings Across Models.
CLIP and NegCLIP lack separation between afrmative and negated
captions. ConCLIP treats all negated captions as identical, regardless of the object type, while the Sentence Transformer shows more ideal
separability along both 'object type' and 'negation' dimensions.
lapping embeddings for afrmative and negated captions in
CLIP
and
NegCLIP
suggest that these models do not dis-
tinguish between positive and negative statements, possi-
bly due to a ﬁbag-of-wordsﬂ shortcut strategy [
10
,
 48
] that
overlooks negation words. This explains why both models
incorrectly select the Negation template, which negates pos-
itive objects, in Figure
 5
.
CoNCLIP
separates positive and
negative captions but fails to distinguish between negative
6


captions of different objects, collapsing all negative caption
embeddings toward a single point (red circle).
We include the embeddings of a text-only Sentence
Transformer [
33
] as a reference that effectively differenti-
ates afrmative and negated captions along distinct ﬁobject
typeﬂ and ﬁnegationﬂ axes, exemplifying ideal separation.
Hybrid captions reveal more evidence of collapsed em-
beddings.
Figure
 6b
 extends the prev ious analysis to hy-
brid captions that combine afrmations and negations. It
provides further evidence that
ConCLIP
employs a shortcut
strategy for embedding linguistic negation, with hybrid and
negated captions collapsing towards a single point (green
circle), indicating signicant compression along the nega-
tion axis. While
CLIP
and
NegCLIP
struggle to distinguish
afrmative from negative statements,
NegCLIP
shows bet-
ter separation for hybrid captions, which appear collapsed
in the CLIP embedding space. This suggests that Neg-
CLIP's poor performance on Hybrid MCQs might be due to
a misalignment between the text and image encoders, rather
than an inability to understand hybrid sentence structure. In
contrast, the
Sentence Transformer
effectively distinguishes
between different caption types and provides semantically
guided representations. For example, it aligns ﬁowers but
not catsﬂ along the line connecting ﬁowersﬂ and ﬁnot cats.ﬂ
5. A Data-Centric Approach for Improving
Negation Understanding
We hypothesize that the tendency of CLIP-based models to
rely on linguistic shortcuts, which hinders their negation un-
derstanding as explored in Section
 4.1
, stems from training
data limitations. In CLIP, training data lacks examples with
explicit negation, leaving it unable to distinguish negated
and afrmed concepts. In contrast, ConCLIP's training data
overts to a single hybrid linguistic template, limiting its
ability to generalize across varied negation structures. Next,
we explore data-centric strategies to address these gaps, in-
troducing a dataset that includes diverse negation examples
spanning a range of linguistic styles.
5.1. Synthesizing auning Negation Dataset
We augment the CC12M dataset [
4
], which contains ap-
proximately 10 million image-text pairs, to generate two
synthetic datasets with negation: CC12M-NegCap and
CC12M-NegMCQ. Our goal is to expose models to a wide
variety of negation scenarios and improve their ability to en-
code negated statements. The process follows these steps:
1.
Object Extraction
: Using LLaMA 3.1 [
6
], we extract
positive objects (those mentioned in the caption) and
negative objects (contextually relevant but not present)
from each image-caption pair in CC12M.
2.
Visual Verication
: An open-vocabulary object detec-
tor [
25
] veries the presence of positive objects and en-
sures the absence of the negative objects in the image.
This step is crucial to avoid introducing incorrect nega-
tions that could confuse the model.
3.
Caption Generation
: For each image, we generate mul-
tiple new caption s that incorporate negated ob jects into
the original captions. LLaMA 3.1 is used to ensure the
generated captions are natural-sounding and reect real-
istic negation scenarios found in retrieval queries.
We construct two variants of the synthetic dataset.
CC12M-NegCap
includes three captions per image with
incorporated negated objects, totaling approximately 30
million captions.
CC12M-NegMCQ
includes four cap-
tions per image: one correct and three hard negatives based
on ob ject annotations, offering stronger training signals
for ne-grained negation understanding and resulting in
around 40 million captions. To balance broad retrieval with
ne-grained negation capabilities, we introduce
CC12M-
NegFull
, a comprehensive dataset that combines CC12M-
NegCap and CC12M-NegMCQ. We will release the ex-
tracted object annotations for each image in CC12M, along
with the corresponding URLs, and all the generated cap-
tions in CC12M-NegFull. This will help the community
build on our dataset and advance research in negation un-
derstanding and multimodal retrieval.
5.2.uning with Data
Standard CLIP Objective on CC12M-NegCap.
Let
B
cap
=
f
(
I
i
; T
i
)
g
N
i
=1
represent a batch of
N
image-caption
pairs from CC12M-NegCap, where each image
I
i
is paired
with a caption
T
i
that describes present and absent objects
in the image. For each batch
B
cap
, we compute a similar-
ity matrix
S
2
R
N

N
, where each element
S
j;k
represents
the cosine similarity between the
j
-th image and the
k
-th
caption. The CLIP objective applies a symmetric cross-
entropy loss over this matrix, encouraging high similarity
for correct image-caption pairs and low similarity for incor-
rect pairs. This loss is denoted as
L
CLIP
(
B
cap
)
and provides
the model with diverse negation examples in a contrastive
learning setup.
Multiple-Choice Objective on CC12M-NegMCQ.
Let
B
mcq
=
f
(
I
i
;
f
T
i;
1
; : : : ; T
i;C
g
)
g
M
i
=1
be a batch of
M
examples from CC12M-NegMCQ, where each image
I
i
is
paired with
C
captions
f
T
i;j
g
C
j
=1
. One caption correctly
describes the image, while the others serve as hard nega-
tives. For our experiments, we set
C
= 4
. To ne-tune
on CC12M-NegMCQ, we compute the cosine similarity be-
tween each image and its four caption options, generating a
set of logits for each image-option pair.
The multiple-choice loss
L
MCQ
(
B
mcq
)
is then computed
by applying a cross-entropy loss over the logits, with the
correct answer index as the target. This loss encourages the
model to assign higher similarity to the correct caption and
lower similarity to the hard negative captions:
7


L
MCQ
(
B
mcq
) =

1
M
M
X
i
=1
log
exp(
logits
i;c
i
)
P
C
j
=1
exp(
logits
i;j
)
;
(1)
where
c
i
indicates the index of the correct caption de-
scribing the
i
-th image.
Combined Training Objective.
The nal objective com-
bines the contrastive loss on CC12M-NegCap with the
MCQ loss on CC12M-NegMCQ, weighted by

to balance
their contributions. The total loss for one batch is:
L
Total
=

L
CLIP
(
B
cap
) + (1


)
L
MCQ
(
B
mcq
)
:
(2)
Evaluation Protocol.
To assess the impact of our data-
centric approach, we ne-tune two pretrained models
(OpenAI CLIP and NegCLIP) on CC12M-NegCap us-
ing the contrastive loss
L
CLIP
. Additionally, we ne-
tune both models on the combined CC12M-NegCap and
CC12M-NegMCQ datasets using
L
Total
in Equation (
2
).
For comparison, we ne-tune these models on the origi-
nal CC12M dataset to isolate the effect of our negation-
enriched datasets. Our goal is to demonstrate that CLIP
models can signicantly improve their understanding of
negation with the right data.
We evaluate the models on two tasks: (i) text-to-image
and text-to-video retrieval on COCO and MSR-VTT, both
with and without negated queries, and (ii) image-to-text and
video-to-text MCQ tasks, where models select the correct
caption from four options. The results are shown in Table
 1
.
Results.
Fine-tuning CLIP and NegCLIP on CC12M-
NegCap leads to signicant improvements in handling
negated queries in retrieval. On COCO, CLIP's R-Neg@5
score increases by 10%, while the gap between R@5 and
R-Neg@5 narrows from 6.8% to 0.7%, indicating that the
netuned model performs nearly as well on negated queries
as on standard ones. A similar pattern is seen in MSR-VTT.
However, ne-tuning on CC12M-NegCap alone does not
improve performance on the MCQ task, suggesting that the
contrastive objective is insufcient for learning ne-grained
negation understanding. To address this, we ne-tune CLIP
and NegCLIP on the combined CC12M-NegFull dataset
using Equation (
2
), yielding substantial improvements on
MCQ tasks. On COCO-MCQ, for instance, NegCLIP's ac-
curacy rises from 10.2% to 51.0%, a 40.8% increase.
Ablation: Effect of varying

.
The table below shows the
impact of varying the weight factor

in the combined loss
L
Total
=

L
CLIP
+ (1


)
L
MCQ
when ne-tuning CLIP
on CC12M-NegFull. As

increases, more weight is placed
on the original CLIP contrastive objective, while a lower

emphasizes the MCQ loss. Properly tuning

is important to
balance between ne-grained MCQ and standard retrieval.

0 0.5 0.9 0.99 1
COCO Recall@5 (%) 33.9 37.3 47.6 54.2 58.5
COCO MCQ Acc (%) 61.0 54.7 50.5 46.9 14.7
Model Fine-tune data R@5 (
"
) R-Neg@5 (
"
) MCQ (
"
)
CLIP
None
 54.8
 48.0
 16.3
CC12M 58.8 54.5 11.2 (
#
5.1)
CC12M-
NegCap 58.5 57.8 14.7
(
#
1.6)
CC12M-
NegFull 54.2 51.9 46.9
(
"
30.6
)
NegCLIP
None
 68.7
 64.4
 10.2
CC12M 70.2 66.0 10.6 (
"
0.4)
CC12M-
NegCap 68.6 67.5 12.5
(
"
2.3
)
CC12M-
NegFull 69.0 67.0 51.0
(
"
40.8
)
(a) COCO Evaluation
Model Fine-tune data R@5 (
"
) R-Neg@5 (
"
) MCQ (
"
)
CLIP
None
 50.6
 45.8
 20.1
CC12M 53.7 49.9 16.9 (
#
3.2)
CC12M-
NegCap 54.1 53.5 20.1
(0.0)
CC12M-
NegFull 46.9 43.9 35.6
(
"
15.5
)
NegCLIP
None
 53.7
 51.0
 15.3
CC12M 56.4 52.6 16.8 (
"
1.5
)
CC12M-
NegCap 56.5 54.6 18.9
(
"
3.6
)
CC12M-
NegFull 54 51.5 36.6
(
"
21.3
)
(b) MSR-VTT Evaluation
Table 1.
Comparison of ne-tuning datasets
on performance
metrics across COCO and MSR-VTT, ne-tuned on respective
datasets and evaluated on retrieval and MCQs. Differences in
MCQ accuracy from the baseline are shown, with increases of
 +1
or more highlighted. Fine-tuning on negation-enriched data sig-
nicantly improves negation understanding (R-Neg and MCQ).
6. Discussion and Conclusions
Implications.
Our ndings point to two broader impli-
cations for enhancing language understanding in VLMs.
From a data perspective, pretraining datasets should include
a diverse array of language constructs, especially those in-
volving nuanced expressions like negation or complex syn-
tactic structures, to help models capture the subtleties of hu-
man language. Currently, many VLMs are pretrained on
datasets that primarily consist of straightforward, afrma-
tive statements, which might limit the models' ability to
understand more subtle language elements. From a learn-
ing perspective, our results suggest that contrastive learn-
ing alone may not be sufcient for ne-grained language
distinctions. We experimented with different values of

in Equation (
2
), which revealed a tradeoff in performance:
higher values improved coarse-grained retrieval but dimin-
ished performance on ne-grained multiple-choice ques-
tions. This suggests that alternative or supplementary train-
ing objectives beyond contrastive learning could enhance
models' sensitivity to nuanced language, enabling more ro-
bust applications in real-world settings where precise lan-
guage interpretation is essential.
Summary.
This paper introduces
NegBench
to systemati-
cally evaluate negation understanding in VLMs. Our nd-
ings reveal that CLIP-based models exhibit a strong afrma-
tion bias, limiting their application in scenarios where nega-
tion is critical, such as medical diagnostics and safety moni-
toring. Through synthetic negation data, we offer a promis-
ing path toward more reliable models. While our synthetic
data approach improves negation understanding, challenges
remain, particularly with ne-grained negation differences.
8


References
[1]
 Hassan Abu Alhaija, Siva Karthik Mustikovela, Lars
Mescheder, Andreas Geiger, and Carsten Rother. Aug-
mented reality meets computer vision: Efcient data gen-
eration for urban driving scenes.
International Journal of
Computer Vision
, 126:961Œ972, 2018.
 2
[2]
 Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Al-
berto Del Bimbo. Effective conditioned and composed im-
age retrieval combining clip-based features. In
Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition
, pages 21466Œ21474, 2022.
 1
,
 2
[3]
 Santiago Castro and Fabian Caba. FitCLIP: Rening large-
scale pretrained image-text models for zero-shot video un-
derstanding tasks. In
33rd British Machine Vision Con-
ference 2022, BMVC 2022, London, UK, November 21-24,
2022
. BMVA Press, 2022.
 2
,
 4
[4]
 Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu
Soricut. Conceptual 12M: Pushing web-scale image-text
pre-training to recognize long-tail visual concepts. In
CVPR
,
2021.
 7
[5]
 Yuhua Chen, Wen Li, Xiaoran Chen, and Luc Van Gool.
Learning semantic segmentation from synthetic data: A geo-
metrically guided input-output adaptation approach. In
Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition
, pages 1841Œ1850, 2019.
 2
[6]
 Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Ab-
hishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil
Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The
llama 3 herd of models.
arXiv preprint arXiv:2407.21783
,
2024.
 3
,
 7
[7]
 Mark Everingham, Luc Van Gool, Christopher KI Williams,
John Winn, and Andrew Zisserman. The pascal visual object
classes (voc) challenge.
International journal of computer
vision
, 88, 2010.
 3
[8]
 Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan
Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten,
Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Or-
gad, Rahim Entezari, Giannis Daras, Sarah M Pratt, Vivek
Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen
Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna,
Pang Wei Koh, Olga Saukh, Alexander Ratner, Shuran
Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont,
Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon,
Vaishaal Shankar, and Ludwig Schmidt. Datacomp: In
search of the next generation of multimodal datasets. In
NeurIPS Datasets and Benchmarks Track
, 2023.
 4
[9]
 Iker Garc
´
a-Ferrero, Bego
Ÿ
na Altuna, Javier Alvez, Itziar
Gonzalez-Dios, and German Rigau. This is not a dataset: A
large negation benchmark to challenge large language mod-
els. In
EMNLP
. Association for Computational Linguistics,
2023.
 2
[10]
 Robert Geirhos, J
¨
orn-Henrik Jacobsen, Claudio Michaelis,
Richard Zemel, Wieland Brendel, Matthias Bethge, and Fe-
lix A Wichmann. Shortcut learning in deep neural networks.
Nature Machine Intelligence
, 2(11), 2020.
 6
[11]
 Hasan Abed Al Kader Hammoud, Hani Itani, Fabio Pizzati,
Philip Torr, Adel Bibi, and Bernard Ghanem. Synthclip: Are
we ready for a fully synthetic clip training?
arXiv preprint
arXiv:2402.01832
, 2024.
 2
[12]
 Laurence R. Horn.
A Natural History of Negation
. University
of Chicago Press, 1989.
 1
[13]
 Wisdom Ikezogwo, Saygin Seyoglu, Fatemeh Ghezloo,
Dylan Geva, Fatwir Sheikh Mohammed, Pavan Kumar
Anand, Ranjay Krishna, and Linda Shapiro. Quilt-1m: One
million image-text pairs for histopathology.
Advances in
neural information processing systems
, 36, 2024.
 2
[14]
 Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Sil-
viana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad
Haghgoo, Robyn Ball, Katie Shpanskaya, et al. Chexpert:
A large chest radiograph dataset with uncertainty labels and
expert comparison. In
Proceedings of the AAAI conference
on articial intelligence
, pages 590Œ597, 2019.
 3
[15]
 Ali Jahanian, Xavier Puig, Yonglong Tian, and Phillip Isola.
Generative models as a data source for multiview representa-
tion learning. In
International Conference on Learning Rep-
resentations
, 2022.
 2
[16]
 Michael P Jordan. The power of negation in english: Text,
context and relevance.
Journal of pragmatics
, 29(6), 1998.
 1
[17]
 Miren Itziar Laka Mugarza.
Negation in syntaxŒon the na-
ture of functional categories and projections
. PhD thesis,
Massachusetts Institute of Technology, 1990.
 3
[18]
 Yehao Li, Yingwei Pan, Ting Yao, and Tao Mei. Compre-
hending and ordering semantics for image captioning. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition
, pages 17990Œ17999, 2022.
 1
,
2
[19]
 Zhengxin Li, Wenzhe Zhao, Xuanyi Du, Guangyao Zhou,
and Songlin Zhang. Cross-modal retrieval and semantic re-
nement for remote sensing image captioning.
Remote Sens-
ing
, 16(1):196, 2024.
 1
,
 2
[20]
 Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Doll
´
ar, and C Lawrence
Zitnick. Microsoft coco: Common objects in context. In
ECCV
, 2014.
 3
[21]
 Ming Y Lu, Bowen Chen, Drew FK Williamson, Richard J
Chen, Ivy Liang, Tong Ding, Guillaume Jaume, Igor
Odintsov, Long Phi Le, Georg Gerber, et al. A visual-
language foundation model for computational pathology.
Nature Medicine
, 30:863Œ874, 2024.
 1
,
 2
,
 4
[22]
 Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei,
Nan Duan, and Tianrui Li. CLIP4Clip: An empirical study
of clip for end to end video clip retrieval.
arXiv preprint
arXiv:2104.08860
, 2021.
 2
[23]
 Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and
Jimmy Lin. Fine-tuning llama for multi-stage text retrieval.
In
Proceedings of the 47th International ACM SIGIR Confer-
ence on Research and Development in Information Retrieval
,
pages 2421Œ2425, 2024.
 1
[24]
 Zixian Ma, Jerry Hong, Mustafa Omer Gul, Mona Gandhi,
Irena Gao, and Ranjay Krishna. Crepe: Can vision-language
foundation models reason compositionally? In
CVPR
, 2023.
1
,
 2
9


[25]
 Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim
Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh
Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran
Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil
Houlsby. Simple open-vocabulary object detection with vi-
sion transformers.
ECCV
, 2022.
 4
,
 7
,
 15
[26]
 Roser Morante and Eduardo Blanco. Recent advances in
processing negation.
Natural Language Engineering
, 27(2):
121Œ130, 2021.
 1
[27]
 Partha Mukherjee, Youakim Badr, Shreyesh Doppalapudi,
Satish M Srinivasan, Raghvinder S Sangwan, and Rahul
Sharma. Effect of negation in sentences on sentiment analy-
sis and polarity detection.
Procedia Computer Science
, 185:
370Œ379, 2021.
 1
[28]
 Medhini Narasimhan, Anna Rohrbach, and Trevor Darrell.
Clip-it! language-guided video summarization.
Advances
in neural information processing systems
, 34:13988Œ14000,
2021.
 2
[29]
 Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, and Jimmy
Lin. Multi-stage document ranking with bert.
arXiv preprint
arXiv:1910.14424
, 2019.
 1
[30]
 Suzanne Petryk, Lisa Dunlap, Keyan Nasseri, Joseph Gon-
zalez, Trevor Darrell, and Anna Rohrbach. On guiding vi-
sual attention with language specication. In
Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition
, pages 18092Œ18102, 2022.
 1
,
 2
[31]
 Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn-
ing transferable visual models from natural language super-
vision. In
ICML
. PMLR, 2021.
 2
,
 4
[32]
 Yongming Rao, Wenliang Zhao, Guangyi Chen, Yansong
Tang, Zheng Zhu, Guan Huang, Jie Zhou, and Jiwen Lu.
Denseclip: Language-guided dense prediction with context-
aware prompting. In
Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition
, pages
18082Œ18091, 2022.
 1
,
 2
[33]
 Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence
embeddings using siamese bert-networks. In
EMNLP
. Asso-
ciation for Computational Linguistics, 2019.
 7
[34]
 Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj
¨
orn Ommer. High-resolution image syn-
thesis with latent diffusion models. In
Conference on Com-
puter Vision and Pattern Recognition (CVPR)
, pages 10684Œ
10695, 2022.
 4
[35]
 Aneeshan Sain, Ayan Kumar Bhunia, Pinaki Nath Chowd-
hury, Subhadeep Koley, Tao Xiang, and Yi-Zhe Song. Clip
for all things zero-shot sketch-based image retrieval, ne-
grained or not. In
Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition
, pages 2765Œ
2775, 2023.
 1
,
 2
[36]
 Christoph Schuhmann, Romain Beaumont, Richard Vencu,
Cade Gordon, Ross Wightman, Mehdi Cherti, Theo
Coombes, Aarush Katta, Clayton Mullis, Mitchell Worts-
man, et al. Laion-5b: An open large-scale dataset for training
next generation image-text models.
Advances in Neural In-
formation Processing Systems
, 35:25278Œ25294, 2022.
 5
[37]
 Christoph Schuhmann, Romain Beaumont, Richard Vencu,
Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo
Coombes, Aarush Katta, Clayton Mullis, Mitchell Worts-
man, Patrick Schramowski, Srivatsa R Kundurthy, Katherine
Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia
Jitsev. LAION-5b: An open large-scale dataset for train-
ing next generation image-text models. In
Thirty-sixth Con-
ference on Neural Information Processing Systems Datasets
and Benchmarks Track
, 2022.
 4
[38]
 Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal,
Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt
Keutzer. How much can clip benet vision-and-language
tasks? In
International Conference on Learning Representa-
tions
.
 1
,
 2
[39]
 Hengcan Shi, Munawar Hayat, Yicheng Wu, and Jianfei
Cai. Proposalclip: Unsupervised open-category object pro-
posal generation via exploiting clip cues. In
Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition
, pages 9611Œ9620, 2022.
[40]
 Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Cliport:
What and where pathways for robotic manipulation. In
Con-
ference on robot learning
, pages 894Œ906. PMLR, 2022.
 1
,
2
[41]
 Jaisidh Singh, Ishaan Shrivastava, Mayank Vatsa, Richa
Singh, and Aparna Bharati. Learnﬂ noﬂ to sayﬂ yesﬂ bet-
ter: Improving vision-language models via negations.
arXiv
preprint arXiv:2403.20312
, 2024.
 1
,
 2
,
 4
[42]
 Yonglong Tian, Lijie Fan, Phillip Isola, Huiwen Chang, and
Dilip Krishnan. Stablerep: Synthetic images from text-to-
image models make strong visual representation learners. In
NeurIPS
, 2023.
 2
[43]
 Yonglong Tian, Lijie Fan, Kaifeng Chen, Dina Katabi, Dilip
Krishnan, and Phillip Isola. Learning vision from mod-
els rivals learning vision from data. In
Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition
, pages 15887Œ15898, 2024.
 2
[44]
 Ekin Tiu, Ellie Talius, Pujan Patel, Curtis P Langlotz, An-
drew Y Ng, and Pranav Rajpurkar. Expert-level detection
of pathologies from unannotated chest x-ray images via self-
supervised learning.
Nature Biomedical Engineering
, 6(12):
1399Œ1406, 2022.
 2
[45]
 Thinh Hung Truong, Timothy Baldwin, Karin Verspoor, and
Trevor Cohn. Language models are not naysayers: an anal-
ysis of language models on negation benchmarks. In
Pro-
ceedings of the 12th Joint Conference on Lexical and Com-
putational Semantics (* SEM 2023)
, pages 101Œ114, 2023.
2
[46]
 Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large
video description dataset for bridging video and language. In
Proceedings of the IEEE conference on computer vision and
pattern recognition
, pages 5288Œ5296, 2016.
 3
[47]
 Jianhao Yuan, Jie Zhang, Shuyang Sun, Philip Torr, and Bo
Zhao. Real-fake: Effective training data synthesis through
distribution matching. In
The Twelfth International Confer-
ence on Learning Representations
, 2024.
 2
[48]
 Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri,
Dan Jurafsky, and James Zou. When and why vision-
10


language models behave like bags-of-words, and what to do
about it? In
ICLR
, 2023.
 2
,
 4
,
 6
,
 12
[49]
 Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu,
Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao,
Mu Wei, Naveen Valluri, et al. Biomedclip: a multimodal
biomedical foundation model pretrained from fteen million
scientic image-text pairs.
arXiv preprint arXiv:2303.00915
,
2023.
 1
,
 2
,
 4
11


Appendix
A. Evaluating LLaVA on NegBench MCQs
In the main paper, we proposed a novel evaluation paradigm
for negation understanding, aimed at simulating real-world
scenarios as closely as possible. We then proceeded to eval-
uate joint embedding-based VLMs, particularly CLIP mod-
els, which are the dominant models for multimodal retrieval
tasks, in addition to being popular for text-to-image gen-
eration, image captioning, and medical multimodal tasks.
However, we recognize that there are other VLMs that can
be useful in certain settings. In particular, instruction-tuned
VLMs like LLaVA open up the path for conversational
VLM chatbots. In this section, we evaluate LLaVA on
the three natural image MCQ tasks in NegBench (COCO,
VOC2007, and HardNeg-Syn). The results are in Figure
 7
.
Figure 7.
Caption.
LLaVA, an instruction-tuned VLM, demonstrates im-
provement.
Figure
 7
 shows that LLaVA signicantly out-
performs CLIP models on the MCQ-Neg tasks. This is par-
ticularly notable because LLaVA uses a CLIP ViT-L/14 vi-
sion encoder, which we have shown in Figure
 4
 to struggle
with negation. The key advantage of LLaVA might be in
its use of the Vicuna LLM for text encoding. Unlike CLIP,
which is pretrained on vision-language pairs that predom-
inantly contain afrmative image captions, LLMs like Vi-
cuna are trained on diverse textual corpora that include both
afrmations and negations. This broader expo sure allows
LLaVA to better interpret negated statements. Additionally,
LLaVA uses a learned projection layer to align vision and
language representations, in contrast to CLIP's contrastive
learning objective, which tends to ignore word order and
subtle linguistic cues like negation [
48
]. We further explore
these differences in Figure
 8
.
Limitations of LLaVA as a retrieval system.
While
LLaVA demonstrates improved negation understanding, it
has signicant limitations as a retrieval model compared
to CLIP. CLIP learns a joint image-text embedding space,
making it highly efcient for retrieval tasks by simply em-
bedding both images and texts, and then computing cosine
similarities. In contrast, LLaVA processes a single image-
text pair at a time and generates text output, which makes
image-to-text retrieval feasible only if all possible captions
can t into the model's context window. For MCQ-Neg, we
applied this method by presenting the image alongside all
possible captions and prompting LLaVA to select the cor-
rect o ne. However, this approach does not scale well with
a large number of candidates and is not applicable for text-
to-image retrieval, where tting all dataset images into the
context window is impractical. Therefore, advancing mod-
els like CLIP is crucial for real-world multimodal retrieval
with negation. In the paper, we explored the data-centric
reasons behind CLIP's failures in negation understanding
and proposed synthetic data strategies to address them.
B. A Closer Look at VLM Negation Failures
To better understand the negation failures of VLMs, we
further analyze the models' tendency to select specic
template types when answering multiple-choice questions
(MCQs) and prov ide further analysis into the embedding
space of these models.
B.1. Template Selection Frequency
Figure
 8
 analyzes the frequency with which different mod-
els select specic template types (Afrmation, Negation,
Hybrid) when answering multiple-choice questions, regard-
less of the correct answer. This analysis helps to reveal po-
tential biases in model behavior and understand why mod-
els may struggle with negation. As shown in Figure
 5
 from
the paper, most models perform poorly on Negation MCQs,
reecting a general struggle with negation understanding.
Figure 8. Template selection frequency for various models on
COCO and VOC2007 datasets, broken down by template type (Af-
rmation, Negation, Hybrid).
Figure
 8
 shows that most CLIP-based models tend to
overselect the Negation template, even when the correct an-
swer is an afrmation or a hybrid statement. This behavior
12


Figure 9. PCA projections of caption embeddings for various CLIP models and the Sentence Transformer. Each point represents a caption
embedding. This gure complements Figure
 6
 by providing a broader view of embedding separation across different VLMs.
is consistent with the observation in Figure
 5
 that most mod-
els perform poorly on Negation MCQs, where they must se-
lect a caption that does not reference the positive object. In
contrast, ConCLIP overselects the Hybrid template, which
combines afrmation and negation in the same statement.
This bias likely stems from ConCLIP's training data, which
overts to a single hybrid linguistic template.
On the other hand, the LLaVA model exhibits a more bal-
anced template selection frequency, indicating that it is less
susceptible to the biases observed in CLIP-based models.
B.2. Template Embedding Analysis
This subsection provides further details about the embed-
ding analysis presented in Figure
 6
 of the main paper. We
13


achieve this by:
1.
 Specifying templates used to generate the embeddings.
2.
 Expanding the embedding analysis to more models.
To generate the embeddings for the PCA projections, we
used ve categories of templates: Afrmation (single ob-
ject), Negation (single object), Afrmation (two objects),
Hybrid (one object afrmed, one negated), and Double
Negation (two objects negated). Each category contains
24 templates, except for Afrmation (two objects) which
has 23. The templates vary sentence structure and wording
while maintaining the same core meaning.
Ł
Afrmation (single object):
24 templates. Examples:
ﬂThis image includes Aﬂ, ﬂA is present in this imageﬂ,
ﬂThis image shows Aﬂ, ﬂA is depicted in this imageﬂ, ﬂA
appears in this imageﬂ.
Ł
Negation (single object):
24 templates. Examples: ﬂThis
image does not include Aﬂ, ﬂA is not present in this im-
ageﬂ, ﬂThis image lacks Aﬂ, ﬂA is not depicted in this
imageﬂ, ﬂA does not appear in this imageﬂ.
Ł
Afrmation (two objects):
23 templates. Examples:
ﬂThis image includes A and Bﬂ, ﬂA and B are present in
this imageﬂ, ﬂThis image shows A and Bﬂ, ﬂA and B are
depicted in this imageﬂ, ﬂA and B appear in this imageﬂ.
Ł
Hybrid (one object afrmed, one negated):
24 tem-
plates. Examples: ﬂThis image includes A but not Bﬂ,
ﬂA is present in this image but not Bﬂ, ﬂThis image shows
A but not Bﬂ, ﬂThis image features A but not Bﬂ, ﬂA ap-
pears in this image but not Bﬂ.
Ł
Double Negation (two objects negated):
24 templates.
Examples: ﬂThis image includes neither A nor Bﬂ, ﬂNei-
ther A nor B are present in this imageﬂ, ﬂThis image
shows neither A nor Bﬂ, ﬂNeither A nor B are depicted
in this imageﬂ, ﬂNeither A nor B appear in this imageﬂ.
While Figure 6 focused on CLIP, NegCLIP, and Con-
CLIP, Figure
 9
 presents an additional visualization with
PCA projections for other CLIP models (varying in size and
pretraining datasets). This broader analysis will provide a
more comprehensive view o f how different CLIP models
handle negation in the embedding space.
C. Dataset and Task Summary for NegBench
We provide a summary of the datasets and tasks used in
NegBench, a framework designed to evaluate Visual Lan-
guage Models (VLMs) on their understanding of negation
across different modalities, including images, videos, and
medical imaging. The benchmark includes both retrieval
and multiple-choice question (MCQ) tasks, with two varia-
tions: templated and LLM-paraphrased. For synthetic data,
we generate 10,000 images using Stable Diffusion, which
serve as hard negatives for one another, enabling a more
focused evaluation of negation comprehension in text-to-
image retrieval tasks.
Each dataset contributes to either Retrieval-Neg or
MCQ-Neg tasks, except for CheXpert, which has two dis-
tinct tasks (Afrmation Control and Negation Understand-
ing) in both MCQ and binary classication formats. Ad-
ditionally, we utilize original retrieval captions for COCO
(5,000) and MSR-VTT (1,000), expanding the overall
dataset size. VOC2007 does not include a Retrieval-Neg
task as it lacks retrieval-style captions.
The total number of task variations across all datasets
in NegBench is 18, and the total number of samples across
all tasks and variations is 79,239. Table
 2
 summarizes the
datasets, tasks, task versions, and sizes.
Ł
COCO
: 5,000 retrieval captions and 5,914 MCQ ques-
tions, resulting in 10,000 retrieval problems and 11,828
MCQ problems with templated and LLM-paraphrased
variations.
Ł
VOC2007
: 5,032 MCQ questions, leading to 10,064 total
samples. No retrieval task is provided due to the absence
of retrieval-style captions.
Ł
MSR-VTT
: 1,000 retrieval captions and 1,000 MCQ
questions, resulting in 2,000 samples per task, including
both variations.
Ł
CheXpert
: Two MCQ tasks (4-choice) and two binary
classication tasks. The 4-choice MCQ covers 690 sam-
ples for afrmation and 1,587 for negation, while the bi-
nary tasks each include 690 samples.
Ł
HardNeg-Syn
: 10,000 synthetic images, used to create
20,000 retrieval and 20,000 MCQ problems across tem-
plated and LLM-paraphrased versions.
14


Table 2.
Summary of datasets and tasks in NegBench.
Each task includes both templated and LLM-paraphrased versions, except for
CheXpert tasks, which are templated only due to their straightforwardness (they directly evaluate diagnostic capabilities in the presence
of negation words). The HardNeg-Syn dataset contains 10,000 synthetic images as hard negatives, offering a more targeted evaluation of
negation understanding. The total number of task variations is 18, with a total of 79,239 samples across all tasks and variations.
Dataset Task Templated LLM-Paraphrased Task Size Notes
COCO
Retrieval-Neg
X X
10,000 Image retrieval with negated captions.
MCQ-Neg
X X
11,828 MCQ task with afrmative, negated, and hybrid options.
VOC2007
MCQ-Neg
X X
10,064 MCQ task. No Retrieval-Neg for VOC2007.
MSR-VTT
Retrieval-Neg
X X
2,000 Video retrieval task with negated captions.
MCQ-Neg
X X
2,000 Video-based MCQ task with temporal context.
CheXpert
(4-choice)
Afrmation Control MCQ
X
Œ 690 Medical image MCQ with 4 choices.
Negation Understanding MCQ
X
Œ 1,587 MCQ task with negation.
CheXpert
(binary)
Afrmation Control
X
Œ 690 Binary classication of medical images.
Negation Understanding
X
Œ 690 Binary classication, negated statements.
HardNeg-Syn
Retrieval-Neg
X X
20,000 Synthetic image retrieval task.
MCQ-Neg
X X
20,000 MCQ task for synthetic images with 4 answer choices.
C.1. Details of Construction
Object Label Selection
We gather a wide range of object text labels from existing datasets like ImageNet.
Scene Description
For each selected object label (
A
), LLaMA 3.1 generates:
A
f
background
description
g
and a related object
f
B
g
, crafting realistic scene contexts.
Image Generation
Using Stable Diffusion, we generate pairs of images:
Positive Image:
f
background
description
g
with
f
A
g
next to
f
B
g
.
Negative Image:
f
background
description
g
with
f
A
g
, excluding
f
B
g
in the negative prompt to
ensure its absence.
Verication
We use OWL-ViT [
25
] to verify the presence and absence of
A
and
B
.
Caption Generation
Captions are generated using templates and paraphrased with LLaMA 3.1 for naturalness.
D. Visualizing the NegBench Evaluation Tasks
In Figures Figures
 10
 to
 13
, we visualize a few samples
from the NegBench retrieval and MCQ tasks we introduced
in the paper. We note that the datasets are diverse in terms
of the nature of visual domain and real-world applicability.
15


Figure 10. Examples of COCO and VOC2007 tasks, including Retrieval with negated captions and MCQ with negation.
Figure 11. Examples of CheXpert MCQ tasks, including the Afrmation Control task and the Negation task.
16


Figure 12. Examples of HardNeg-Syn (MCQ-Neg) tasks. Images in this dataset are constructed in pairs, with each pair differing by a
single object (the cactus in the rst pair), making the dataset particularly suitable for studying negation understanding.
17


Figure 13. Examples of MSR-VTT tasks, including Retrieval-Neg (with negated captions about a complex water rescue scene) and MCQ-
Neg (with answer choices about the presence or absence of actions like walking).
18
