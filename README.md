# Framework for training and evaluating dialogue agents

To get started, you should create your own "datasets" and "results" directories and download the appropriate data

### __Description__
Execute - shell scripts that kick off run.py with arguments filled in
Run - starting point for all training and evaluation
Model - major modules that fall into one of three categories:
  * Preprocess - functions used to preprocess data
      - preparing data by embedding them and tokenization
      - collecting data through crowdsourcing platforms
  * Learn - modules used to build dialogue models
      - belief tracking
      - policy management
      - response generation
      - encoders and decoders
  * Evaluate - functions used for qualitative and quantitative evaluation
      - qualitiative evaluation with text
      - quantitative measures like BLEU or accuracy or loss
  * components.py - single file of shared model components used by others

Utils - various functionality that is resuable
  * external: most code written was imported from elsewwhere
  * internal: most code written was developed interally

Scripts - one-off methods that are sparingly used

### __Order_of_Execution__
  1. Utils should always load first since they are used everywhere
  2. Components will load next since they are shared across modules
  3. Preprocess, Learn and Evaluate modules import components
  4. Run.py comes last, it will inherit modules to perform its tasks
As a result, utils should have very few imports, and components should only import from utils and never from modules

### __Development_Strategy__
The process is that most work probably starts off as random scripts. If the same script is being used often, or if certain functions are being repeated, then they probably deserve to be placed into utils or components.  If there are a collection of utils that clearly fall into a theme, they should get placed into a module within the model folder.
