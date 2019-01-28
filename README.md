# Framework for training and evaluating dialogue agents

To get started, you should create your own "datasets" and "results" directories and download the appropriate data. Training has worked before with Woz2, DSTC2, Stanford In-Car dataset, Maluuba Frames Corpus, MultiWoz, FB bAbI Dialog, and Microsoft E2E Dialogue Challenge. Although a handful of these now are deprecated.  Next, there are execution shell scripts in the `execute` folder which can be run for various purposes.

Much of the code is in flux, so please feel free to make pull requests or leave suggestions!

### Major Concepts
Every experiment or trial is considered a system.  Systems have different versions and the system in production should always be the latest stable version.
Each system is made up of objects and operators.
  * Objects - the building blocks of the system.  They are the things that get acted on and transformed.  Typical examples are the Dataset and Model.  The dataset is often made up of Dialogue objects, which are in turn made up of Utterance objects.  End-to-End Models are often made up of Intent Tracker, Knowledge Base Operator, Policy Manager and Text Generator modules.
     - blocks: these are the most basic building blocks that most other items import
     - models: these hold some of the more typical models seen in papers, such as a Transformer or Seq2Seq
     - modules: these wrap around a model and offer semantic meaning, such as a Intent Tracker wrapped around a GLAD model
  * Operators - the main actors in the system.  They are the things that take action and transform objects.  These items are divided by the phase of operation.
     - preprocessing: loading and pre-processing data
     - learning: trainining models
     - evaluation: running qualitative, quantitative, and visual evaluation (ie. plottting)
  * Session - a session is a special type of object since there is a time component involved.  A system might spin up two sessions to interact with each other, such as a agent session and user session.  These sessions might last indefinitely, and a system might have multiple concurrent sessions running at once. (To be developed)

#### Belief Tracking
Arguably the most important task of a dialogue agent is to understand the user intent.  Intents are broken down into five pieces:
  1. domain
  2. dialogue act
  3. slot
  4. relation
  5. value

The vast majority of systems assume that the domain and sub-domain are given and consequently are ignored during prediction, instead focusing on only 'act(slot=value)'.  (See Victor's GLAD model and Hannah's xIntent from Event2Mind)

#### Policy Management
The policy manager is formalized as a POMDP where the latent state is the underlying user intent.  The actual dialogue state is also composed of other information such as dialogue context, turn count and complete semantic frame.  Agent actions are defined by the ontology for any given domain.  Clarification is studied by expanding the question types to include:
  1. conventional clarification request - what did you want? I understood almost nothing
  2. partial clarification requests - what was the area you mentioned? I understood one piece of information
  3. confirmation through mention of alternatives - did you say the north part of town? I only misunderstood one piece of information
  4. reformulation of information - so basically you want asian food, right? I understood almost everything

In conjunction with the policy manager is the user simulator which makes training an RL agent feasible (Xiujin's D3Q stuff).

#### Text Generation
Currently, template based mechanisms or straight-forward LSTM decoders are used for response generation.  This will be expanded in the future to include ability to add in diversity, persona (Jiwei's papers) and other desirable traits based on Grice's maxims (Ari, Jan, Max and others)

### Order of Execution
  1. Utils should always load first since they are used everywhere
  2. Objects will load next since they are shared across Operators
  3. Operator modules are found in Preprocess, Learn and Evaluate
  4. Run.py is the true starting point, it will import Operators to perform tasks
  5. The system will be chosen which will perform the actual operations
