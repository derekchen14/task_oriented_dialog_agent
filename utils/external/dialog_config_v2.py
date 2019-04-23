starting_dialogue_acts = ['open', 'request']

inform_slots = ['city', 'date', 'genre', 'moviename', 'numberofpeople', 'starttime', 'state', 'theater']
request_slots = ['other', 'theater', 'ticket', 'unknown']

#   A Basic Set of Feasible actions to be Consdered By an RL agent
feasible_agent_actions = [
    {'dialogue_act':"greeting", 'inform_slots':{}, 'request_slots':{}},
    {'dialogue_act': "confirm_question", 'inform_slots': {}, 'request_slots': {}},
    {'dialogue_act': "confirm_answer", 'inform_slots': {}, 'request_slots': {}},
    {'dialogue_act': "thanks", 'inform_slots': {}, 'request_slots': {}},
    {'dialogue_act': "deny", 'inform_slots': {}, 'request_slots': {}},
]
#   A Basic Set of Feasible actions to be Consdered By an User
feasible_user_actions = [
    {'dialogue_act': "thanks", 'inform_slots': {}, 'request_slots': {}},
    {'dialogue_act': "deny", 'inform_slots': {}, 'request_slots': {}},
    {'dialogue_act': "closing", 'inform_slots': {}, 'request_slots': {}},
    {'dialogue_act': "confirm_answer", 'inform_slots': {}, 'request_slots': {}}
]

for inf_slot in inform_slots:
    feasible_agent_actions.append({'dialogue_act': 'inform', 'inform_slots': {inf_slot: "PLACEHOLDER"}, 'request_slots': {}})
    feasible_user_actions.append({'dialogue_act': 'inform', 'inform_slots': {inf_slot: "PLACEHOLDER"}, 'request_slots': {}})
for req_slot in request_slots:
    feasible_agent_actions.append({'dialogue_act': 'request', 'inform_slots': {}, 'request_slots': {req_slot: "<unk>"}})
    feasible_user_actions.append({'dialogue_act': 'request', 'inform_slots': {}, 'request_slots': {req_slot: "<unk>"}})
feasible_agent_actions.append({'dialogue_act': 'inform', 'inform_slots': {"task": "complete"}, 'request_slots': {}})


# Dialog status
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
NO_OUTCOME_YET = 0

# Rewards
SUCCESS_REWARD = 40
FAILURE_REWARD = 0
PER_TURN_REWARD = 0

#  Special Slot Values
I_DO_NOT_CARE = "I do not care"
NO_VALUE_MATCH = "NO VALUE MATCHES!!!"
TICKET_AVAILABLE = 'Ticket Available'

#  Constraint Check
CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1
nlg_beam_size = 10

#  run_mode: 0 for dia-act; 1 for NL; 2 for no output; 3 for skip everything
run_mode = 3
auto_suggest = False # (or True)
