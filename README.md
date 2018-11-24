# Framework for training and evaluating dialogue agents

To get started, you should create your own "datasets" and "results" directories and download the appropriate data

__Description__
Components - modules used to build dialogue models, including belief tracking, policy management, and response generation
Utils - various functionality that is resuable
  external: most code written was imported from elsewwhere
  internal: most code written was developed interally
Scripts - one-off methods that are sparingly used

The process is that most work probably starts off as random scripts. If the same script is being used often, or if certain functions are being repeated, then they probably deserve to be placed into utils.  If there are a collection of utils that clearly fall into a theme, they should get placed into a class or module, and placed somewhere in components folder.

Modules within components fall into one of three categories:
  * Preprocess - functions used to preprocess and prepare data, including collection of data through crowdsourcing platforms
  * Learn - functions used for building and training models
  * Evaluate - functions used for qualitative and quantitative evaluation