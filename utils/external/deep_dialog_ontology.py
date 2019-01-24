# movie
movie_sys_request_slots = ['moviename', 'theater', 'starttime', 'date', 'numberofpeople', 'genre', 'state', 'city', 'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor', 'description', 'other', 'numberofkids']
movie_sys_inform_slots = ['moviename', 'theater', 'starttime', 'date', 'genre', 'state', 'city', 'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor', 'description', 'other', 'numberofkids', 'taskcomplete', 'ticket']
movie_user_request_slots = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
movie_user_inform_slots = ['moviename', 'theater'] #, 'starttime'
# restaurant
restaurant_sys_request_slots = ['address', 'atmosphere', 'choice', 'city', 'cuisine', 'date', 'distanceconstraints', 'dress_code', 'food', 'mealtype', 'numberofpeople', 'numberofkids', 'occasion', 'personfullname', 'rating', 'restaurantname', 'restauranttype', 'seating', 'starttime', 'state', 'zip', 'other']
restaurant_sys_inform_slots = ['address', 'atmosphere', 'choice', 'city', 'closing', 'cuisine', 'date', 'distanceconstraints', 'dress_code', 'food', 'mealtype', 'numberofpeople', 'numberofkids', 'occasion', 'personfullname', 'phonenumber', 'pricing', 'rating', 'restaurantname', 'restauranttype', 'seating', 'starttime', 'state', 'zip', 'reservation', 'taskcomplete', 'other', 'greeting']
restaurant_user_request_slots = ["restaurantname", "date", "numberofpeople", "starttime", "address"]
restaurant_user_inform_slots = ["restaurantname", "address"] # "starttime", "date",
# taxi
taxi_sys_request_slots = ['taxi', 'dropoff_location', 'cost', 'pickup_location', 'taxi_company', 'city', 'zip', 'pickup_location_city', 'state', 'other', 'numberofpeople', 'pickup_time', 'dropoff_location_city', 'date', 'car_type', 'name', 'distanceconstraints']
taxi_sys_inform_slots = ['taxi', 'dropoff_location', 'cost', 'pickup_location', 'taskcomplete', 'taxi_company', 'city', 'zip', 'pickup_location_city', 'state', 'other', 'numberofpeople', 'pickup_time', 'dropoff_location_city', 'date', 'car_type', 'name', 'distanceconstraints', 'greeting', 'closing']
taxi_user_request_slots = ["pickup_location", "dropoff_location", "date", "numberofpeople", "pickup_time", "car_type", "taxi"]
taxi_user_inform_slots = ["car_type", "cost"] #

start_dia_acts = {'request':['moviename', 'starttime', 'theater', 'city', 'state', 'date', 'genre', 'ticket', 'numberofpeople']}
relations = ["=", "<", ">", "!="]
dialogue_acts = ["open", "close", "request", "inform", "accept", "reject", "question", "answer", "acknow", "confuse"]

################################################################################
# Dialog status
################################################################################
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
NO_OUTCOME_YET = 0
# Rewards
SUCCESS_REWARD = 50
FAILURE_REWARD = 0
PER_TURN_REWARD = 0

CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1
nlg_beam_size = 10
################################################################################
#  Special Slot Values
################################################################################
I_DO_NOT_CARE = "I do not care"
NO_VALUE_MATCH = "NO VALUE MATCHES!!!"
TICKET_AVAILABLE = 'Ticket Available'
################################################################################
#  run_mode: 0 for dia-act; 1 for NL; 2 for no output
################################################################################
run_mode = 0
auto_suggest = 0

################################################################################
#   A Basic Set of Feasible actions to be Consdered By an RL agent
################################################################################
feasible_actions = [
    #   greeting actions
    {'diaact':"greeting", 'inform_slots':{}, 'request_slots':{}},
    #   confirm_question actions
    {'diaact':"confirm_question", 'inform_slots':{}, 'request_slots':{}},
    #   confirm_answer actions
    {'diaact':"confirm_answer", 'inform_slots':{}, 'request_slots':{}},
    #   thanks actions
    {'diaact':"thanks", 'inform_slots':{}, 'request_slots':{}},
    #   deny actions
    {'diaact':"deny", 'inform_slots':{}, 'request_slots':{}},
]

for slot in movie_sys_inform_slots:
    feasible_actions.append({'diaact':'inform', 'inform_slots':{slot:"PLACEHOLDER"}, 'request_slots':{}})
for slot in movie_sys_request_slots:
    feasible_actions.append({'diaact':'request', 'inform_slots':{}, 'request_slots': {slot: "UNK"}})
