# Legend for the Diagnostic S, T, and C Labels

This file describes the labels used in the CSV to classify the prerequisite abilities required by the VSV questions.

A single question may receive multiple labels because it may simultaneously require spatial, temporal, and causal abilities.

---

## S Labels — Spatial Reasoning

### S0 — Recognition of the involved entities
Requires the model to correctly identify the people, objects, animals, or locations needed to answer the question.

**Example:** recognizing the dog and its owner before determining where the dog is located.

### S1 — Simple spatial relation
Requires the model to identify a spatial relation that is directly observable at a given moment.

Typical relations include:
- above / below;
- inside / outside;
- in front of / behind;
- to the right / to the left;
- near / far;
- on / in;
- origin or destination of a movement.

**Example:** determining where an object is located or on which surface it is placed.

### S2 — Change in spatial relation over time
Requires the model to compare the position of an entity at different moments in the video.

**Example:** determining where an object was before being picked up and where it is afterwards.

### S3 — Spatial relation conditioned by an event or temporal phase
Requires the model to identify a spatial relation that is valid only during or after a specific event.

**Example:** determining where the child is when she starts crying.

---

## T Labels — Temporal Reasoning

### T0 — Recognition of events
Requires the model to correctly identify the actions or events mentioned in the question.

**Example:** recognizing the moment when a person sits down and the moment when the same person greets someone.

### T1 — Temporal localization
Requires the model to determine when an event occurs in the video.

This may refer to:
- a timestamp;
- a specific moment;
- a delimited phase of the video.

**Example:** determining when a person starts running.

### T2 — Order or relation between events
Requires the model to compare two events and determine their temporal relation.

Typical relations include:
- before;
- after;
- during;
- simultaneously;
- while;
- already in progress.

**Example:** determining whether the woman sighs before or after the phone rings.

### T3 — Global position within the video
Requires the model to determine whether an event occurs:
- at the beginning;
- in the middle;
- near the end of the video.

**Example:** determining whether an action occurs at the beginning or at the end of the scene.

### T4 — Duration
Requires the model to estimate or determine how long an event, activity, or interval lasts.

**Example:** determining how long it takes to complete a procedure.

### T5 — Change of state
Requires the model to recognize a transformation between an initial state and a final state.

**Example:** an object is initially closed and later opened, or a person changes from standing to sitting.

---

## C Labels — Causal Reasoning

### C0 — Recognition of the effect
Requires the model to identify the result, consequence, or state that must be explained.

**Example:** recognizing that a person is crying, becoming angry, or falling.

### C1 — Recognition of a candidate cause
Requires the model to identify the event, action, or condition that may have produced the effect.

**Example:** recognizing that a child starts crying after falling.

### C2 — Temporal relation between cause and effect
Requires the model to verify that the cause precedes or accompanies the effect in a coherent way.

**Example:** determining that the fall occurs before the crying.

### C3 — Causal link
Requires the model to establish that one event does not merely precede another, but explains it.

**Example:** concluding that the child is crying because she fell.

### C4 — Implicit causal inference
Requires the model to reconstruct a cause that is not fully visible or explicitly stated, using contextual evidence from the video.

**Example:** inferring why a person is worried from the surrounding situation and their reactions.

### C5 — Necessity or counterfactual reasoning
Requires the model to evaluate what would have happened if a cause had not occurred or if a condition had been different.

**Example:** determining whether an effect would still have occurred in the absence of the observed cause.

---

## Interpreting the Compositionality Lists

The `composizionalita` column contains a list of the dimensions involved in each question.

Examples:

- `["spatial"]`
- `["temporal"]`
- `["causal"]`
- `["spatial", "temporal"]`
- `["temporal", "causal"]`
- `["spatial", "temporal", "causal"]`

The presence of more than one dimension indicates that the question requires a combination of abilities.

**Example:**  
“Where is the child when she starts crying?” requires:
- spatial recognition of the child's location;
- temporal recognition of the onset of crying.

---

## Interpreting the Complexity Level

The `livello_complessita_1_5` column provides an overall estimate of the expected difficulty of the question based on its linguistic formulation and required prerequisites.

- **1 — very low:** direct recognition of an entity or a simple relation;
- **2 — low:** an explicit relation or a single easily localized event;
- **3 — medium:** comparison between events, change of state, or a short inference;
- **4 — high:** integration of multiple events, relations, or dimensions;
- **5 — very high:** implicit inference, complex causal reasoning, or a spatial-temporal-causal combination.

---

## Methodological Note

The labels describe the prerequisite abilities required by the question. They do not directly measure the actual difficulty of the video.

The real difficulty may also depend on:
- video quality and duration;
- number of people or objects;
- occlusions;
- speed of the events;
- audio clarity;
- temporal distance between events;
- scene ambiguity;
- amount of implicit knowledge required.
