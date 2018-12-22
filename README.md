# Framework for training and evaluating dialogue agents

To get started, you should create your own "datasets" and "results" directories and download the appropriate data

### Major Concepts
Every experiment or trial is considered a system.  Systems have different versions and the system in production should always be the latest stable version.
Each system is made up of objects and operators.
  * Objects - the building blocks of the system.  They are the things that get acted on and transformed.  Typical examples are the Dataset and Model.  The dataset is often made up of Dialogue objects, which are in turn made up of Utterance objects.  End-to-End Models are often made up of Intent Tracker, Knowledge Base Looker, Policy Manager and Text Generator objects.
  * Operators - the main actors in the system.  They are the things that take action and transform objects.  Preprocessors, Batchers, and Dataloaders typically operate on datasets.  Builders, Learners and Evaluators typically operate on models.
  * Session - a session is a special type of object since there is a time component involved.  A system might spin up two sessions to interact with each other, such as a agent session and user session.  These sessions might last indefinitely, and a system might have multiple concurrent sessions running at once.

Arguably the most important task of a dialogue agent is to understand the user intent.  Intents are broken down into five pieces:
  1. domain
  2. sub-domain
  3. dialogue act
  4. slot
  5. value
The vast majority of systems assume that the domain and sub-domain are given and consequently are ignored during prediction, instead focusing on only 'act(slot=value)'.

### Directory Description
Execute - shell scripts that kick off run.py with arguments filled in.  The kinds of execution scripts are typically organized by the types of system to run.

Run - starting point for all systems
  * Preprocess - functions used to preprocess data
      - preparing data by embedding them and tokenization
      - collecting data through crowdsourcing platforms
  * Learn - modules used to build dialogue models
      - builder: these assemble modules for:
        - belief tracking
        - policy management
        - response generation
      - these are made up of: encode, decode, attend, embed, and transform modules
      - learner: with train and validation components
      - user simulator
  * Evaluate - functions used for qualitative and quantitative evaluation
      - qualitiative evaluation with text
      - quantitative measures like BLEU or accuracy or loss
  * components.py - single file of shared model components used by others, such as running inference and variable creation

Utils - various functionality that is resuable
  * external: most code written was imported from elsewwhere
  * internal: most code written was developed interally

Scripts - one-off methods that are sparingly used

### Order of Execution
  1. Utils should always load first since they are used everywhere
  2. Components will load next since they are shared across modules
  3. Preprocess, Learn and Evaluate modules import components
  4. Run.py comes last, it will inherit modules to perform its tasks

As a result, utils should have very few imports, and components should only import from utils and never from modules
